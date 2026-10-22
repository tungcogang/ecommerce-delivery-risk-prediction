import pandas as pd

# Đọc dataset
df = pd.read_csv("../data/Train.csv")

# Hiển thị 5 dòng đầu tiên
print(df.head())

# Kích thước dataset
print("\nKích thước dataset:")
print(df.shape)

# Thông tin các cột
print("\nThông tin dataset:")
print(df.info())