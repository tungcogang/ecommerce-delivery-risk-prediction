
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score
)


# 1. Đường dẫn
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Train.csv"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

print("Kích thước dữ liệu:", df.shape)


# 3. Khai báo các cột
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


# 4. Tạo bộ tiền xử lý
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_cols),
        ("numerical", numerical_pipeline, numerical_cols)
    ]
)


# 5. Chia train/test một lần duy nhất
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# 6. Ưu tiên Recall cho lớp giao trễ (nhãn 1)
recall_late_scorer = make_scorer(
    recall_score,
    pos_label=1
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 7. Khai báo không gian tham số
param_distributions = {
    "decision_tree": {
        "classifier__max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
        "classifier__min_samples_split": [2, 5, 10, 20, 40],
        "classifier__min_samples_leaf": [1, 2, 5, 10, 20, 30],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__class_weight": [None, "balanced"]
    },

    "random_forest": {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [None, 6, 10, 12, 16, 20, 30],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"]
    }
}


# 8. Hai mô hình cần tinh chỉnh
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )
}


# 9. Tinh chỉnh từng mô hình
results = []

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print("ĐANG TINH CHỈNH:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions[
            "decision_tree" if model_name == "Decision Tree"
            else "random_forest"
        ],
        n_iter=25,
        scoring=recall_late_scorer,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nTham số tốt nhất:")
    print(search.best_params_)

    print(
        f"\nRecall lớp giao trễ tốt nhất trong CV: "
        f"{search.best_score_:.4f}"
    )

    print(f"Accuracy trên test: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Giao đúng hạn (0)", "Giao trễ (1)"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print("Thứ tự hàng/cột: [Giao đúng hạn (0), Giao trễ (1)]")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    file_name = (
        "decision_tree_tuned.joblib"
        if model_name == "Decision Tree"
        else "random_forest_tuned.joblib"
    )

    model_path = MODELS_DIR / file_name
    joblib.dump(best_model, model_path)

    print("\nĐã lưu mô hình:", model_path)

    results.append({
        "Mô hình": model_name,
        "Recall giao trễ CV": search.best_score_,
        "Accuracy test": accuracy_score(y_test, y_pred),

        "Recall giao trễ test": recall_score(
        y_test, y_pred, pos_label=1
),
        "F1 giao trễ test": classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
        )["1"]["f1-score"],

    })


# 10. Tổng hợp kết quả
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("TỔNG KẾT SAU TINH CHỈNH")
print("=" * 60)

print(
    results_df[
        [
            "Mô hình",
            "Recall giao trễ CV",
            "Accuracy test",
            "Recall giao trễ test",
            "F1 giao trễ test"
        ]
    ].to_string(index=False)
)

results_df.to_csv(
    MODELS_DIR / "tuning_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu bảng kết quả:", MODELS_DIR / "tuning_results.csv")
