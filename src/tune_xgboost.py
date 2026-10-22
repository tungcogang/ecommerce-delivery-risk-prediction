import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================
# 1. ĐƯỜNG DẪN VÀ ĐỌC DỮ LIỆU
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "Train.csv"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=["ID", "Reached.on.Time_Y.N"]
)

# Quy ước nhãn gốc:
# 0 = Giao trễ
# 1 = Giao đúng hạn
y = df["Reached.on.Time_Y.N"].astype(int)

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

print("===== THÔNG TIN DỮ LIỆU =====")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

print(
    "Nhãn train:",
    y_train.value_counts().sort_index().to_dict()
)

print(
    "Nhãn test:",
    y_test.value_counts().sort_index().to_dict()
)


# ==========================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================

categorical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

numerical_pipeline = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

preprocessor = ColumnTransformer(transformers=[
    (
        "categorical",
        categorical_pipeline,
        categorical_cols
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_cols
    )
])


# ==========================================
# 3. XÂY DỰNG PIPELINE XGBOOST
# ==========================================

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),

    (
        "classifier",
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=1
        )
    )
])


# ==========================================
# 4. KHÔNG GIAN THAM SỐ
# ==========================================

param_distributions = {
    "classifier__n_estimators": [
        150, 250, 350, 500
    ],

    "classifier__max_depth": [
        2, 3, 4, 5, 6
    ],

    "classifier__learning_rate": [
        0.02, 0.03, 0.05, 0.08, 0.1
    ],

    "classifier__min_child_weight": [
        1, 3, 5, 7
    ],

    "classifier__subsample": [
        0.7, 0.8, 0.9, 1.0
    ],

    "classifier__colsample_bytree": [
        0.7, 0.8, 0.9, 1.0
    ],

    "classifier__gamma": [
        0, 0.1, 0.3, 0.5
    ],

    "classifier__reg_lambda": [
        1, 3, 5, 10
    ],

    "classifier__scale_pos_weight": [
        1.0, 1.3, 1.5, 2.0
    ]
}


# ==========================================
# 5. CROSS-VALIDATION VÀ TÌM THAM SỐ
# ==========================================

# Tối ưu Recall cho lớp 0 = giao trễ.
# Giữ nguyên nhãn gốc 0 và 1 trong toàn bộ quá trình.

recall_late_scorer = make_scorer(
    recall_score,
    pos_label=0,
    zero_division=0
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=recall_late_scorer,
    cv=cv,
    refit=True,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    error_score="raise"
)

print("\n===== BẮT ĐẦU TÌM THAM SỐ =====")

# Quan trọng: huấn luyện trực tiếp với y_train,
# không chuyển y_train thành True/False.
search.fit(X_train, y_train)

print("\n===== KẾT QUẢ TÌM KIẾM =====")

print(
    "Recall giao trễ trung bình CV:",
    round(search.best_score_, 4)
)

print("Tham số tốt nhất:")
print(search.best_params_)


# ==========================================
# 6. ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================

best_model = search.best_estimator_

# Dự đoán giữ nguyên nhãn gốc:
# 0 = giao trễ
# 1 = giao đúng hạn
y_pred = best_model.predict(X_test)

print("\n===== KIỂM TRA NHÃN =====")

print(
    "Các lớp của mô hình:",
    best_model.named_steps["classifier"].classes_
)

print(
    "Nhãn dự đoán:",
    pd.Series(y_pred).value_counts().sort_index().to_dict()
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision_late = precision_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall_late = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

f1_late = f1_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

print("\n===== XGBOOST ĐÃ TINH CHỈNH =====")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision giao trễ (0): {precision_late:.4f}")
print(f"Recall giao trễ (0): {recall_late:.4f}")
print(f"F1-score giao trễ (0): {f1_late:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Giao trễ (0)",
            "Giao đúng hạn (1)"
        ],
        zero_division=0
    )
)

print("===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(cm)

print("\nCách đọc ma trận:")
print("Hàng: nhãn thực tế")
print("Cột: nhãn dự đoán")
print("Thứ tự lớp: [0 = giao trễ, 1 = giao đúng hạn]")


# ==========================================
# 7. LƯU MÔ HÌNH VÀ KẾT QUẢ
# ==========================================

model_path = (
    MODELS_DIR / "xgboost_tuned_pipeline.joblib"
)

joblib.dump(best_model, model_path)

results_path = (
    MODELS_DIR / "xgboost_tuning_results.csv"
)

results_df = pd.DataFrame(search.cv_results_)

results_df = results_df.sort_values(
    "rank_test_score"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n===== LƯU THÀNH CÔNG =====")
print("Mô hình:", model_path)
print("Kết quả tìm kiếm:", results_path)
