# pip install pandas numpy tensorflow tensorflowjs

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

CSV_PATH = "asl_raw_2026-05-22T17-33-53.csv"  

df = pd.read_csv(CSV_PATH)

coord_cols = []
for i in range(21):
    coord_cols += [f"x{i}", f"y{i}", f"z{i}"]

landmarks = df[coord_cols].values.astype(np.float32)
landmarks = landmarks.reshape(-1, 21, 3)
labels = df["letter"].values

# Preprocessing
features = []
for lm in landmarks:
    # Translate
    lm = lm - lm[0]
    # Scale
    scale = np.linalg.norm(lm[9]) or 1.0
    lm = lm / scale
    # Flatten
    features.append(lm[1:].flatten())

X = np.vstack(features).astype(np.float32)

# Encode labels
classes = sorted(np.unique(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_int = np.array([class_to_idx[c] for c in labels], dtype=np.int32)

# Save label map
label_map = {i: c for i, c in enumerate(classes)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("✅ Saved label_map.json")

# Augmentation
def augment(batch, noise_std=0.02, scale_jitter=0.05):
    noise = np.random.normal(0, noise_std, batch.shape)
    scale = 1.0 + np.random.uniform(-scale_jitter, scale_jitter, size=(batch.shape[0], 1))
    return (batch * scale) + noise

X_aug = [X]
y_aug = [y_int]
for _ in range(3):
    X_aug.append(augment(X))
    y_aug.append(y_int)

X = np.vstack(X_aug)
y_int = np.hstack(y_aug)
print(f"✅ Augmented dataset: {len(X)} samples")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_int, test_size=0.2, random_state=42, stratify=y_int
)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test accuracy: {acc:.4f}")

# Export
import json
import os

os.makedirs('asl_tfjs_model', exist_ok=True)

with open('asl_tfjs_model/model_arch.json', 'w') as f:
    f.write(model.to_json())

weights = model.get_weights()
manifest = []
for i, w in enumerate(weights):
    fname = f'weight_{i}.bin'
    w.astype(np.float32).tofile(f'asl_tfjs_model/{fname}')
    manifest.append({
        "name": fname,
        "shape": list(w.shape),
        "dtype": "float32"
    })

with open('asl_tfjs_model/weights_manifest.json', 'w') as f:
    json.dump(manifest, f)

print("Saved TF.js compatible files to ./asl_tfjs_model/")