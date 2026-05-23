# pip install pandas numpy tensorflow

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

CSV_PATH = "asl_raw_2026-05-23T18-27-04.csv"

df = pd.read_csv(CSV_PATH)

coord_cols = []
for i in range(21):
    coord_cols += [f"x{i}", f"y{i}", f"z{i}"]

landmarks = df[coord_cols].values.astype(np.float32)
landmarks = landmarks.reshape(-1, 21, 3)
labels = df["letter"].values


def extract_robust_features(lm_hand):
    # Center coordinates relative to the wrist (landmark 0)
    lm_centered = lm_hand - lm_hand[0]

    # Normalize scale using wrist-to-middle-finger-knuckle distance (0 -> 9)
    scale = np.linalg.norm(lm_centered[9]) or 1.0
    lm_scaled = lm_centered / scale

    # Structural distances capture fist tightness and finger curl
    finger_tips = [4, 8, 12, 16, 20]
    knuckles = [2, 5, 9, 13, 17]

    distances = []
    for tip in finger_tips:
        distances.append(np.linalg.norm(lm_scaled[tip]))
    for tip, knuck in zip(finger_tips, knuckles):
        distances.append(np.linalg.norm(lm_scaled[tip] - lm_scaled[knuck]))

    # Orientation vectors capture hand rotation and pitch
    palm_normal = np.cross(lm_scaled[5] - lm_scaled[0], lm_scaled[17] - lm_scaled[0])
    palm_normal /= (np.linalg.norm(palm_normal) or 1.0)

    hand_direction = lm_scaled[9] - lm_scaled[0]
    hand_direction /= (np.linalg.norm(hand_direction) or 1.0)

    feat = np.concatenate([
        lm_scaled.flatten(),
        np.array(distances),
        palm_normal,
        hand_direction
    ])
    return feat


features = [extract_robust_features(lm) for lm in landmarks]
X = np.vstack(features).astype(np.float32)

INPUT_SHAPE = X.shape[1]

classes = sorted(np.unique(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_int = np.array([class_to_idx[c] for c in labels], dtype=np.int32)

label_map = {i: c for i, c in enumerate(classes)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("Saved label_map.json")


def augment(batch, noise_std=0.015, scale_jitter=0.03):
    noise = np.random.normal(0, noise_std, batch.shape)
    scale = 1.0 + np.random.uniform(-scale_jitter, scale_jitter, size=(batch.shape[0], 1))
    return (batch * scale) + noise


X_aug = [X]
y_aug = [y_int]
for _ in range(4):
    X_aug.append(augment(X))
    y_aug.append(y_int)

X = np.vstack(X_aug)
y_int = np.hstack(y_aug)
print(f"Augmented dataset: {len(X)} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(INPUT_SHAPE,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=200,
    batch_size=64,
    callbacks=[es],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

# Native TF.js converter with strict layer-naming alignment
import os
import json

export_dir = 'asl_tfjs_model'
os.makedirs(export_dir, exist_ok=True)

print("\nGenerating web-ready TF.js artifacts manually...")

keras_arch = json.loads(model.to_json())

# Provide ONLY batch_input_shape to satisfy the TF.js layer parser
if "config" in keras_arch and "layers" in keras_arch["config"]:
    first_layer = keras_arch["config"]["layers"][0]
    if first_layer["class_name"] == "InputLayer":
        first_layer["config"].pop("input_shape", None)
        first_layer["config"]["batch_input_shape"] = [None, INPUT_SHAPE]

weight_data = bytearray()
weight_manifest_entries = []

# Skip Keras structural layers that carry no weights (Input, Dropout, etc.)
weight_bearing_layers = [l for l in model.layers if len(l.get_weights()) > 0]

for layer in weight_bearing_layers:
    layer_name = layer.name
    weights = layer.get_weights()

    # Dense layers: [Kernel, Bias]
    if len(weights) == 2:
        w_kernel, w_bias = weights[0], weights[1]

        weight_manifest_entries.append({
            "name": f"{layer_name}/kernel",
            "shape": list(w_kernel.shape),
            "dtype": "float32"
        })
        weight_data.extend(w_kernel.astype(np.float32).tobytes())

        weight_manifest_entries.append({
            "name": f"{layer_name}/bias",
            "shape": list(w_bias.shape),
            "dtype": "float32"
        })
        weight_data.extend(w_bias.astype(np.float32).tobytes())

    # BatchNormalization layers: [Gamma, Beta, Moving Mean, Moving Variance]
    elif len(weights) == 4:
        gamma, beta, mean, variance = weights[0], weights[1], weights[2], weights[3]
        param_names = ["gamma", "beta", "moving_mean", "moving_variance"]

        for w_param, p_name in zip([gamma, beta, mean, variance], param_names):
            weight_manifest_entries.append({
                "name": f"{layer_name}/{p_name}",
                "shape": list(w_param.shape),
                "dtype": "float32"
            })
            weight_data.extend(w_param.astype(np.float32).tobytes())

shard_filename = "group1-shard1of1.bin"
with open(os.path.join(export_dir, shard_filename), "wb") as f:
    f.write(weight_data)

tfjs_model_json = {
    "format": "layers-model",
    "generatedBy": "Native Python Custom Exporter",
    "convertedBy": None,
    "modelTopology": keras_arch,
    "weightsManifest": [
        {
            "paths": [shard_filename],
            "weights": weight_manifest_entries
        }
    ]
}

with open(os.path.join(export_dir, "model.json"), "w") as f:
    json.dump(tfjs_model_json, f, indent=2)

print(f"Successfully created standard TF.js files in ./{export_dir}/")
print(f"   -> model.json")
print(f"   -> {shard_filename}")