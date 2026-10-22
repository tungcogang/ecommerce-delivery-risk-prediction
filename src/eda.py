import pandas as pd

# Đọc dataset
df = pd.read_csv("../data/Train.csv")

print("===== PHÂN BỐ GIAO HÀNG =====")

print(df["Reached.on.Time_Y.N"].value_counts())

print("\n===== TỶ LỆ GIAO HÀNG =====")

print(df["Reached.on.Time_Y.N"].value_counts(normalize=True) * 100)