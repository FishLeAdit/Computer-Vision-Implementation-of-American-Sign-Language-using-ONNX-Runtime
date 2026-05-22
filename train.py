# pip install pandas numpy scikit-learn skl2onnx onnxruntime
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

CSV_PATH = "asl_raw_2026-05-22T17-33-53.csv" 
df = pd.read_csv(CSV_PATH)

# Landmarks into (N, 21, 3) arrays
coord_cols = []
for i in range(21):
    coord_cols += [f"x{i}", f"y{i}", f"z{i}"]

landmarks = df[coord_cols].values.astype(np.float32)  # shape (N, 63)
landmarks = landmarks.reshape(-1, 21, 3)               # shape (N, 21, 3)

labels = df["letter"].values
hands = df["hand"].values if "hand" in df.columns else np.array(["right"] * len(df))

# Canonicalize== Mirroring Left Hand for Right Hand Use
for i in range(len(landmarks)):
    if hands[i] == "left":
        landmarks[i, :, 0] = 1.0 - landmarks[i, :, 0]

# Feature engineering == Translation & Scale Invariance
features = []

for lm in landmarks:
    # Wrist is origin
    lm = lm - lm[0]

    # Wrist 2 mid finger xd
    scale = np.linalg.norm(lm[9])
    if scale < 1e-6:
        scale = 1.0  # fallback

    lm = lm / scale

    # Drop wrist and flatten vector
    vec = lm[1:].flatten() 
    features.append(vec)

X = np.vstack(features).astype(np.float32)
y = labels

# Train / Test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Features per sample: {X.shape[1]}")

# Train Random Forest
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# Eval
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print(f"Test Accuracy: {acc:.4f}")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show confused pairs
print("\nConfusion Matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
print(cm_df)

# Browser inference onyx export
initial_type = [("float_input", FloatTensorType([None, 60]))]
onnx_model = convert_sklearn(
    clf,
    initial_types=initial_type,
    target_opset=12,
    options={id(clf): {"zipmap": False}},  # output raw probabilities
)

onnx_path = "asl_sign_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"\n✅ ONNX model saved: {onnx_path}")

# Onyx runtime sanity check
sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: X_test[:5]})[0]
print("\nONNX Runtime sanity check (first 5 predictions):")
print("  Sklearn :", y_pred[:5])
print("  ONNX    :", pred_onnx)

# Save label mapping for JavaScript frntend
label_map = {i: cls for i, cls in enumerate(clf.classes_)}
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("✅ Label map saved: label_map.json")