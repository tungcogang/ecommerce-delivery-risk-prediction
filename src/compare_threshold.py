# Ưu tiên phát hiện đơn giao trễ (nhãn 1)
from pathlib import Path

import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
final_comparison_rows = []

# ======================================
# 1. Đọc dữ liệu và chia train/test
# ======================================
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

# Các ngưỡng muốn khảo sát
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.4058, 0.45, 0.4830, 0.50]

# Mục tiêu nghiệp vụ tạm thời:
# phát hiện ít nhất 90% số đơn thực sự giao trễ
MIN_RECALL = 0.80

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Decision Tree": MODEL_DIR / "decision_tree_tuned.joblib",
    "Random Forest": MODEL_DIR / "random_forest_tuned.joblib"
}

selected_thresholds = {}

# ======================================
# 2. Khảo sát ngưỡng trên OOF của train
# ======================================
for model_name, model_path in models.items():
    print("\n" + "=" * 65)
    print(f"MÔ HÌNH: {model_name}")
    print("=" * 65)

    model = joblib.load(model_path)

    # Mỗi mẫu train được dự đoán khi nó nằm ở fold validation
    oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )

    class_late_index = list(model.classes_).index(1)
    p_late_oof = oof_proba[:, class_late_index]

    rows = []

    for threshold in THRESHOLDS:
        # Dự đoán giao trễ (nhãn 1) nếu xác suất giao trễ đạt ngưỡng
        y_pred_oof = (p_late_oof >= threshold).astype(int)

        precision_late = precision_score(
            y_train, y_pred_oof,
            pos_label=1, zero_division=0
        )
        recall_late = recall_score(
            y_train, y_pred_oof,
            pos_label=1, zero_division=0
        )
        f1_late = f1_score(
            y_train, y_pred_oof,
            pos_label=1, zero_division=0
        )

        rows.append({
            "Threshold": threshold,
            "Precision_late": precision_late,
            "Recall_late": recall_late,
            "F1_late": f1_late
        })

    results = pd.DataFrame(rows)

    print("\n--- Kết quả OOF trên tập train ---")
    print(results.round(4).to_string(index=False))

    # Chọn ngưỡng chỉ dựa trên OOF:
    # trong các ngưỡng giữ Recall >= 90%,
    # chọn ngưỡng có Precision cao nhất.
    eligible = results[results["Recall_late"] >= MIN_RECALL]

    if eligible.empty:
        print(
            f"\nKhông có ngưỡng nào giữ Recall >= {MIN_RECALL:.0%}."
        )
        continue

    best_row = eligible.sort_values(
        by=["Precision_late", "F1_late", "Threshold"],
        ascending=[False, False, False]
    ).iloc[0]

    selected_threshold = float(best_row["Threshold"])
    selected_thresholds[model_name] = selected_threshold

    print(f"\nNgưỡng chọn từ OOF: {selected_threshold:.4f}")
    print(f"Precision OOF: {best_row['Precision_late']:.4f}")
    print(f"Recall OOF:    {best_row['Recall_late']:.4f}")
    print(f"F1 OOF:        {best_row['F1_late']:.4f}")

    # ======================================
    # 3. Đánh giá đúng một lần trên test
    # ======================================
    # Ngưỡng đã được chọn từ train OOF, không chọn theo test
    test_proba = model.predict_proba(X_test)

    p_late_test = test_proba[:, class_late_index]

    y_pred_test = (p_late_test >= selected_threshold).astype(int)

    # Đánh giá trên tập test ở ngưỡng mặc định và ngưỡng đã chọn
    test_thresholds = sorted(set([0.5, selected_threshold]))

    for threshold in test_thresholds:
        y_pred_test = (p_late_test >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_test)
        precision_late = precision_score(
            y_test, y_pred_test, pos_label=1, zero_division=0
        )
        recall_late = recall_score(
            y_test, y_pred_test, pos_label=1, zero_division=0
        )
        f1_late = f1_score(
            y_test, y_pred_test, pos_label=1, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred_test, labels=[0, 1]
        ).ravel()

        final_comparison_rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision_late": precision_late,
            "Recall_late": recall_late,
            "F1_late": f1_late,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

        print(f"\n{model_name} | Threshold = {threshold:.4f}")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision late : {precision_late:.4f}")
        print(f"Recall late    : {recall_late:.4f}")
        print(f"F1-score late  : {f1_late:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred_test, labels=[0, 1]))

    # Lưu bảng khảo sát OOF để đưa vào báo cáo
    output_csv = MODEL_DIR / (
        model_name.lower().replace(" ", "_") + "_threshold_comparison.csv"
    )
    results.to_csv(output_csv, index=False)

print("\nNgưỡng được chọn:", selected_thresholds)
print("Đã lưu các bảng khảo sát vào thư mục models.")


# Tổng hợp và lưu kết quả so sánh cuối cùng
final_comparison = pd.DataFrame(final_comparison_rows)

# Xác định thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tạo đường dẫn đến thư mục models
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Lưu bảng so sánh
output_path = os.path.join(MODELS_DIR, "final_model_comparison.csv")

final_comparison.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== BẢNG SO SÁNH MÔ HÌNH CUỐI CÙNG =====")
print(final_comparison.round(4).to_string(index=False))
print(f"\nĐã lưu kết quả tại: {output_path}")


