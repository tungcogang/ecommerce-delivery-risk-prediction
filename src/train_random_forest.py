
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. Xác định đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Train.csv"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

print("Kích thước dữ liệu:", df.shape)


# 3. Tách đặc trưng X và nhãn y
X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]


# 4. Phân loại cột dữ liệu
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


# 5. Tiền xử lý cột chữ
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# 6. Tiền xử lý cột số
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# 7. Kết hợp tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_cols),
        ("numerical", numerical_pipeline, numerical_cols)
    ]
)


# 8. Xây dựng Pipeline Random Forest
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# 9. Chia train/test giống các mô hình trước
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 10. Huấn luyện mô hình
print("\nĐang huấn luyện Random Forest...")
model.fit(X_train, y_train)


# 11. Dự đoán
y_pred = model.predict(X_test)


# 12. Đánh giá
print("\n===== KẾT QUẢ RANDOM FOREST =====")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Giao trễ (0)", "Giao đúng hạn (1)"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print("Thứ tự hàng/cột: [Giao trễ (0), Giao đúng hạn (1)]")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# 13. Lưu mô hình
model_path = MODELS_DIR / "random_forest_pipeline.joblib"
joblib.dump(model, model_path)

print("\nĐã lưu mô hình tại:")
print(model_path)
