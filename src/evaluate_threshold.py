
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# =========================
# 1. Đường dẫn và dữ liệu
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Train.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Mục tiêu Precision ban đầu cho lớp giao trễ
TARGET_PRECISION = 0.65

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Decision Tree": MODEL_DIR / "decision_tree_tuned.joblib",
    "Random Forest": MODEL_DIR / "random_forest_tuned.joblib"
}

threshold_results = {}

# =========================
# 2. Phân tích từng mô hình
# =========================
for model_name, model_path in models.items():
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    model = joblib.load(model_path)

    # OOF probabilities: chỉ dùng tập train để chọn ngưỡng
    oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )

    # Tìm đúng cột xác suất thuộc lớp 0 (giao trễ)
    class_zero_index = list(model.classes_).index(0)
    p_late_oof = oof_proba[:, class_zero_index]

    # Xem đường Precision-Recall của lớp giao trễ
    precision, recall, thresholds = precision_recall_curve(
        (y_train == 0).astype(int),
        p_late_oof
    )

    pr_auc = average_precision_score(
        (y_train == 0).astype(int),
        p_late_oof
    )

    print(f"Average Precision (OOF): {pr_auc:.4f}")

    # precision và recall có thêm một phần tử cuối
    # nên chỉ lấy precision[:-1], recall[:-1] để ghép với thresholds
    valid_indices = [
        i for i in range(len(thresholds))
        if precision[i] >= TARGET_PRECISION
    ]

    if valid_indices:
        # Trong các ngưỡng đạt Precision mục tiêu,
        # chọn ngưỡng có Recall cao nhất
        best_index = max(
            valid_indices,
            key=lambda i: recall[i]
        )
        selection_note = (
            f"Ngưỡng đạt Precision >= {TARGET_PRECISION:.0%}"
        )
    else:
        # Nếu không có ngưỡng đáp ứng, tạm chọn F1 cao nhất
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_index = int(f1_values.argmax())
        selection_note = (
            "Không đạt Precision mục tiêu; "
            "tạm chọn ngưỡng có F1 cao nhất"
        )

    threshold = float(thresholds[best_index])

    print(selection_note)
    print(f"Ngưỡng được chọn: {threshold:.4f}")
    print(f"Precision OOF lớp giao trễ: {precision[best_index]:.4f}")
    print(f"Recall OOF lớp giao trễ:    {recall[best_index]:.4f}")

    # Vẽ đường Precision-Recall
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.axhline(
        TARGET_PRECISION,
        linestyle="--",
        label=f"Precision mục tiêu = {TARGET_PRECISION:.0%}"
    )
    plt.scatter(
        recall[best_index],
        precision[best_index],
        label=f"Ngưỡng = {threshold:.3f}"
    )
    plt.xlabel("Recall - lớp giao trễ (0)")
    plt.ylabel("Precision - lớp giao trễ (0)")
    plt.title(f"Precision-Recall: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = MODEL_DIR / (
        model_name.lower().replace(" ", "_") + "_pr_curve.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.show()
    plt.close()

    # =========================
    # 3. Đánh giá cuối trên test
    # =========================
    # Không dùng test để chọn ngưỡng
    test_proba = model.predict_proba(X_test)
    p_late_test = test_proba[:, class_zero_index]

    # Xác suất lớp 0 >= ngưỡng => dự đoán giao trễ (0)
    y_pred = (p_late_test >= threshold).astype(int)
    y_pred = 1 - y_pred  # Đổi True/False thành nhãn 0/1

    print("\n--- Kết quả trên tập test ---")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(
        "Precision lớp giao trễ: "
        f"{precision_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
    )
    print(
        "Recall lớp giao trễ:    "
        f"{recall_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
    )
    print(
        "F1 lớp giao trễ:         "
        f"{f1_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Giao trễ (0)", "Giao đúng hạn (1)"],
        zero_division=0
    ))

    print("Confusion matrix (hàng: thực tế, cột: dự đoán):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    threshold_results[model_name] = {
        "threshold": threshold,
        "target_precision": TARGET_PRECISION,
        "selection_note": selection_note,
        "oof_average_precision": float(pr_auc)
    }

# Lưu ngưỡng để dùng lại khi triển khai mô hình
threshold_path = MODEL_DIR / "decision_thresholds.json"

with open(threshold_path, "w", encoding="utf-8") as f:
    json.dump(threshold_results, f, indent=4, ensure_ascii=False)

print(f"\nĐã lưu ngưỡng tại: {threshold_path}")

