
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. XÁC ĐỊNH ĐƯỜNG DẪN
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Train.csv"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. ĐỌC VÀ TÁCH DỮ LIỆU
# ==========================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

categorical_cols = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Product_importance",
    "Gender"
]

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Kích thước tập train:", X_train.shape)
print("Kích thước tập test:", X_test.shape)

# ==========================================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_pipeline, categorical_cols),
    ("numerical", numerical_pipeline, numerical_cols)
])

# ==========================================
# 4. KHỞI TẠO MÔ HÌNH
# ==========================================

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", gradient_boosting)
])

# ==========================================
# 5. HUẤN LUYỆN
# ==========================================

print("\nĐang huấn luyện Gradient Boosting...")

model.fit(X_train, y_train)

print("Huấn luyện hoàn tất!")

# ==========================================
# 6. DỰ ĐOÁN
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. ĐÁNH GIÁ MÔ HÌNH
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision_late = precision_score(
    y_test, y_pred, pos_label=0, zero_division=0
)

recall_late = recall_score(
    y_test, y_pred, pos_label=0, zero_division=0
)

f1_late = f1_score(
    y_test, y_pred, pos_label=0, zero_division=0
)

print("\n===== KẾT QUẢ GRADIENT BOOSTING =====")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision giao trễ (0): {precision_late:.4f}")
print(f"Recall giao trễ (0): {recall_late:.4f}")
print(f"F1-score giao trễ (0): {f1_late:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Giao trễ (0)", "Giao đúng hạn (1)"],
    zero_division=0
))

print("===== CONFUSION MATRIX =====")

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# ==========================================
# 8. LƯU MÔ HÌNH
# ==========================================

model_path = MODELS_DIR / "gradient_boosting_pipeline.joblib"

joblib.dump(model, model_path)

print("\nĐã lưu mô hình tại:", model_path)
