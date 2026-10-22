
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Train.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "Reached.on.Time_Y.N"

print("===== KIỂM TRA DỮ LIỆU =====")
print("Kích thước:", df.shape)
print("\nSố lượng giá trị thiếu:")
print(df.isnull().sum())

# 2. Tạo bản sao để khảo sát, không làm thay đổi dữ liệu gốc
analysis_df = df.copy()

# 3. Khảo sát tương quan giữa các cột số và nhãn
numeric_cols = analysis_df.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_cols = [
    col for col in numeric_cols
    if col != "ID"
]

correlation = analysis_df[numeric_cols].corr()

print("\n===== TƯƠNG QUAN VỚI NHÃN =====")
print(
    correlation[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Tương quan giữa các biến số")
plt.tight_layout()
plt.show()

# 4. Khảo sát Discount_offered theo tình trạng giao hàng
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=analysis_df,
    x=TARGET,
    y="Discount_offered"
)
plt.title("Discount_offered theo tình trạng giao hàng")
plt.xlabel("Nhãn (0/1)")
plt.ylabel("Discount_offered")
plt.tight_layout()
plt.show()

# 5. Khảo sát Weight_in_gms theo tình trạng giao hàng
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=analysis_df,
    x=TARGET,
    y="Weight_in_gms"
)
plt.title("Weight_in_gms theo tình trạng giao hàng")
plt.xlabel("Nhãn (0/1)")
plt.ylabel("Weight_in_gms")
plt.tight_layout()
plt.show()

# 6. Khảo sát tương tác giữa giảm giá và cân nặng
analysis_df["Discount_group"] = pd.cut(
    analysis_df["Discount_offered"],
    bins=[-float("inf"), 10, 20, float("inf")],
    labels=["Thấp (<=10)", "Trung bình (11-20)", "Cao (>20)"]
)

analysis_df["Weight_group"] = pd.cut(
    analysis_df["Weight_in_gms"],
    bins=[-float("inf"), 2000, 4000, 6000, float("inf")],
    labels=["<=2000g", "2001-4000g", "4001-6000g", ">6000g"]
)

interaction = analysis_df.pivot_table(
    index="Discount_group",
    columns="Weight_group",
    values=TARGET,
    aggfunc="mean",
    observed=False
)


# Đếm số đơn hàng trong từng tổ hợp giảm giá và cân nặng
count_table = analysis_df.pivot_table(
    index="Discount_group",
    columns="Weight_group",
    values=TARGET,
    aggfunc="count",
    observed=False,
    fill_value=0
)

print("\n===== SỐ ĐƠN HÀNG THEO NHÓM =====")
print(count_table)


print("\n===== TỶ LỆ NHÃN 1 THEO GIẢM GIÁ VÀ CÂN NẶNG =====")
print((interaction * 100).round(1))

plt.figure(figsize=(10, 5))
sns.heatmap(
    interaction * 100,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)


count_table = analysis_df.pivot_table(
    index="Discount_group",
    columns="Weight_group",
    values=TARGET,
    aggfunc="count",
    observed=False,
    fill_value=0
)

print("\n===== SỐ ĐƠN HÀNG THEO NHÓM =====")
print(count_table)

plt.title("Tỷ lệ nhãn 1 theo nhóm giảm giá và cân nặng (%)")
plt.xlabel("Nhóm cân nặng")
plt.ylabel("Nhóm giảm giá")
plt.tight_layout()
plt.show()
