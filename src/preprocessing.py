#
#
# # import pandas as pd
# #
# # # 1. Đọc dữ liệu
# # df = pd.read_csv("../data/Train.csv")
# #
# # # 2. Loại bỏ cột ID và tách X, y
# # X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
# # y = df["Reached.on.Time_Y.N"]
# #
# # # 3. Kiểm tra kết quả
# # print("Kích thước dữ liệu ban đầu:", df.shape)
# # print("Kích thước X:", X.shape)
# # print("Kích thước y:", y.shape)
# #
# # print("\nCác cột trong X:")
# # print(X.columns.tolist())
# #
# # print("\nPhân bố của y:")
# # print(y.value_counts())
#
#
#
# import pandas as pd
#
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
#
# # 1. Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # 2. Tách X và y
# X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
# y = df["Reached.on.Time_Y.N"]
#
# # 3. Chia train/test trước khi mã hóa
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )
#
# # 4. Xác định các cột categorical
# categorical_cols = [
#     "Warehouse_block",
#     "Mode_of_Shipment",
#     "Product_importance",
#     "Gender"
# ]
#
# # 5. Tạo bộ mã hóa
# preprocessor = ColumnTransformer(
#     transformers=[
#         (
#             "categorical",
#             OneHotEncoder(handle_unknown="ignore"),
#             categorical_cols
#         )
#     ],
#     remainder="passthrough"
# )
#
# # 6. Học cách mã hóa từ train và biến đổi dữ liệu
# X_train_processed = preprocessor.fit_transform(X_train)
# X_test_processed = preprocessor.transform(X_test)
#
# # 7. Kiểm tra kết quả
# print("X_train ban đầu:", X_train.shape)
# print("X_test ban đầu:", X_test.shape)
#
# print("X_train sau mã hóa:", X_train_processed.shape)
# print("X_test sau mã hóa:", X_test_processed.shape)
#
# print("y_train:", y_train.shape)
# print("y_test:", y_test.shape)



import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# 1. Xác định đường dẫn project
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Train.csv"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

print("Kích thước dữ liệu ban đầu:", df.shape)


# 3. Tách X và y
X = df.drop(columns=["ID", "Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

target = "Reached.on.Time_Y.N"

print("Phân bố nhãn gốc:")
print(df[target].value_counts().sort_index())

print("\nTỷ lệ từng nhãn:")
print(df[target].value_counts(normalize=True).sort_index() * 100)


# 4. Xác định các cột categorical và numerical
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


# 5. Tạo quy trình tiền xử lý cho cột chữ
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# 6. Tạo quy trình tiền xử lý cho cột số
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# 7. Kết hợp hai quy trình
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_cols),
        ("numerical", numerical_pipeline, numerical_cols)
    ]
)


# 8. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Học quy trình tiền xử lý từ train
X_train_processed = preprocessor.fit_transform(X_train)

# Chỉ áp dụng quy trình đã học lên test
X_test_processed = preprocessor.transform(X_test)


# 10. Lưu bộ tiền xử lý
preprocessor_path = MODELS_DIR / "preprocessor.joblib"

joblib.dump(preprocessor, preprocessor_path)


# 11. Kiểm tra kết quả
print("\n===== KẾT QUẢ TIỀN XỬ LÝ =====")

print("X_train ban đầu:", X_train.shape)
print("X_test ban đầu:", X_test.shape)

print("X_train sau tiền xử lý:", X_train_processed.shape)
print("X_test sau tiền xử lý:", X_test_processed.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nPhân bố y_train:")
print(y_train.value_counts())

print("\nPhân bố y_test:")
print(y_test.value_counts())

print("\nĐã lưu bộ tiền xử lý tại:")
print(preprocessor_path)

