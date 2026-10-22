# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Đếm số lượng giao đúng / giao trễ
# counts = df["Reached.on.Time_Y.N"].value_counts()
#
# # Vẽ biểu đồ
# plt.figure(figsize=(7, 5))
#
# sns.barplot(
#     x=counts.index,
#     y=counts.values
# )
#
# plt.title("Phân bố tình trạng giao hàng")
# plt.xlabel("Giao đúng hạn (1) / Giao trễ (0)")
# plt.ylabel("Số lượng đơn hàng")
#plt.show()

#------------------------------------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng/trễ theo mức giảm giá
# delivery_rate = df.groupby("Discount_offered")["Reached.on.Time_Y.N"].mean() * 100
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO MỨC GIẢM GIÁ =====")
# print(delivery_rate)
#
# # Vẽ biểu đồ
# plt.figure(figsize=(10, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo mức giảm giá")
# plt.xlabel("Mức giảm giá")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")

#plt.show()

#----------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng hạn theo trọng lượng
# delivery_rate = df.groupby("Weight_in_gms")["Reached.on.Time_Y.N"].mean() * 100
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO TRỌNG LƯỢNG =====")
# print(delivery_rate)
#
# # Vẽ biểu đồ
# plt.figure(figsize=(10, 5))
#
# plt.plot(
#     delivery_rate.index,
#     delivery_rate.values,
#     marker="o"
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo trọng lượng hàng")
# plt.xlabel("Trọng lượng hàng (g)")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
#
# plt.grid(True)
#
# plt.show()

#-------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Chia trọng lượng thành các nhóm
# df["Weight_group"] = pd.cut(
#     df["Weight_in_gms"],
#     bins=[0, 2000, 3000, 4000, 5000, 6000, 10000],
#     labels=[
#         "< 2000g",
#         "2000-3000g",
#         "3000-4000g",
#         "4000-5000g",
#         "5000-6000g",
#         "> 6000g"
#     ]
# )
#
# # Tính tỷ lệ giao đúng hạn theo nhóm trọng lượng
# delivery_rate = (
#     df.groupby("Weight_group", observed=True)["Reached.on.Time_Y.N"]
#     .mean() * 100
# )
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO NHÓM TRỌNG LƯỢNG =====")
# print(delivery_rate)
#
# # Vẽ biểu đồ
# plt.figure(figsize=(10, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo nhóm trọng lượng")
# plt.xlabel("Nhóm trọng lượng")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
# print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO NHÓM TRỌNG LƯỢNG =====")
# print(df["Weight_group"].value_counts().sort_index())
#
# plt.show()

#----------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng hạn theo phương thức vận chuyển
# delivery_rate = (
#     df.groupby("Mode_of_Shipment")["Reached.on.Time_Y.N"]
#     .mean() * 100
# )
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO PHƯƠNG THỨC VẬN CHUYỂN =====")
# print(delivery_rate)
#
# # Số lượng đơn hàng theo phương thức
# print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO PHƯƠNG THỨC =====")
# print(df["Mode_of_Shipment"].value_counts())
#
# # Vẽ biểu đồ
# plt.figure(figsize=(8, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo phương thức vận chuyển")
# plt.xlabel("Phương thức vận chuyển")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
#
# plt.show()

#-----------------------------------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng hạn theo kho
# delivery_rate = (
#     df.groupby("Warehouse_block")["Reached.on.Time_Y.N"]
#     .mean() * 100
# )
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO KHO =====")
# print(delivery_rate)
#
# # Số lượng đơn hàng theo kho
# print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO KHO =====")
# print(df["Warehouse_block"].value_counts())
#
# # Vẽ biểu đồ
# plt.figure(figsize=(8, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo kho")
# plt.xlabel("Kho")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
#
# plt.show()
#---------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng hạn theo đánh giá khách hàng
# delivery_rate = (
#     df.groupby("Customer_rating")["Reached.on.Time_Y.N"]
#     .mean() * 100
# )
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO ĐÁNH GIÁ KHÁCH HÀNG =====")
# print(delivery_rate)
#
# # Số lượng đơn hàng theo rating
# print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO ĐÁNH GIÁ =====")
# print(df["Customer_rating"].value_counts().sort_index())
#
# # Vẽ biểu đồ
# plt.figure(figsize=(8, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo đánh giá khách hàng")
# plt.xlabel("Đánh giá khách hàng")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
#
# plt.show()

#-------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Đọc dữ liệu
# df = pd.read_csv("../data/Train.csv")
#
# # Tính tỷ lệ giao đúng hạn theo mức độ quan trọng
# delivery_rate = (
#     df.groupby("Product_importance")["Reached.on.Time_Y.N"]
#     .mean() * 100
# )
#
# print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO MỨC ĐỘ QUAN TRỌNG =====")
# print(delivery_rate)
#
# # Số lượng đơn hàng
# print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO MỨC ĐỘ QUAN TRỌNG =====")
# print(df["Product_importance"].value_counts())
#
# # Vẽ biểu đồ
# plt.figure(figsize=(8, 5))
#
# sns.barplot(
#     x=delivery_rate.index,
#     y=delivery_rate.values
# )
#
# plt.title("Tỷ lệ giao đúng hạn theo mức độ quan trọng sản phẩm")
# plt.xlabel("Mức độ quan trọng")
# plt.ylabel("Tỷ lệ giao đúng hạn (%)")
#
# plt.show()

#------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("../data/Train.csv")

# Tính tỷ lệ giao đúng hạn theo giới tính
delivery_rate = (
    df.groupby("Gender")["Reached.on.Time_Y.N"]
    .mean() * 100
)

print("===== TỶ LỆ GIAO ĐÚNG HẠN THEO GIỚI TÍNH =====")
print(delivery_rate)

# Số lượng đơn hàng theo giới tính
print("\n===== SỐ LƯỢNG ĐƠN HÀNG THEO GIỚI TÍNH =====")
print(df["Gender"].value_counts())

# Vẽ biểu đồ
plt.figure(figsize=(7, 5))

sns.barplot(
    x=delivery_rate.index,
    y=delivery_rate.values
)

plt.title("Tỷ lệ giao đúng hạn theo giới tính")
plt.xlabel("Giới tính")
plt.ylabel("Tỷ lệ giao đúng hạn (%)")

plt.show()