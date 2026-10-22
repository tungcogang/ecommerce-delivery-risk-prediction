#
# import os
# import joblib
# import pandas as pd
#
# # =========================
# # 1. XÁC ĐỊNH ĐƯỜNG DẪN
# # =========================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# MODEL_PATH = os.path.join(
#     BASE_DIR, "models", "random_forest_tuned.joblib"
# )
#
# # Ngưỡng được chọn từ bước khảo sát threshold
# THRESHOLD = 0.35
#
#
# # =========================
# # 2. NẠP MÔ HÌNH
# # =========================
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         f"Không tìm thấy mô hình tại: {MODEL_PATH}"
#     )
#
# model = joblib.load(MODEL_PATH)
#
#
# # =========================
# # 3. HÀM DỰ ĐOÁN ĐƠN HÀNG
# # =========================
# def predict_delivery(order_data):
#     """
#     order_data: dictionary chứa thông tin của một đơn hàng.
#     """
#
#     # Chuyển dữ liệu đầu vào thành DataFrame
#     input_df = pd.DataFrame([order_data])
#
#     # Lấy xác suất thuộc lớp giao trễ (nhãn 1)
#     classes = list(model.classes_)
#     late_index = classes.index(1)
#
#     late_probability = model.predict_proba(input_df)[0][late_index]
#
#     # Phân loại theo ngưỡng 0.35
#     prediction = int(late_probability >= THRESHOLD)
#
#     if prediction == 1:
#         status = "CẢNH BÁO: Đơn hàng có nguy cơ giao trễ"
#     else:
#         status = "Đơn hàng được dự đoán giao đúng hạn"
#
#     return {
#         "late_probability": late_probability,
#         "prediction": prediction,
#         "status": status
#     }
#
#
# # =========================
# # 4. CHẠY THỬ VỚI ĐƠN HÀNG MẪU
# # =========================
# if __name__ == "__main__":
#
#     sample_order = {
#         "Warehouse_block": "A",
#         "Mode_of_Shipment": "Ship",
#         "Customer_care_calls": 3,
#         "Customer_rating": 4,
#         "Cost_of_the_Product": 200,
#         "Prior_purchases": 3,
#         "Product_importance": "medium",
#         "Gender": "F",
#         "Discount_offered": 10,
#         "Weight_in_gms": 3500
#     }
#
#     result = predict_delivery(sample_order)
#
#     print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")
#     print(f"Xác suất giao trễ: {result['late_probability']:.2%}")
#     print(f"Ngưỡng phân loại: {THRESHOLD:.0%}")
#     print(f"Kết quả: {result['status']}")



import os
import joblib
import pandas as pd


# ==================================================
# 1. CẤU HÌNH VÀ ĐƯỜNG DẪN
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "random_forest_tuned.joblib"
)

THRESHOLD = 0.35

# Các giá trị phân loại hợp lệ theo dataset
VALID_VALUES = {
    "Warehouse_block": ["A", "B", "C", "D", "E"],
    "Mode_of_Shipment": ["Ship", "Flight", "Road"],
    "Product_importance": ["low", "medium", "high"],
    "Gender": ["F", "M"]
}


# ==================================================
# 2. NẠP MÔ HÌNH
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy mô hình:\n{MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

# Xác định vị trí của nhãn giao trễ = 1
classes = list(model.classes_)

if 1 not in classes:
    raise ValueError("Mô hình không có nhãn giao trễ = 1.")

LATE_INDEX = classes.index(1)


# ==================================================
# 3. HÀM NHẬP DỮ LIỆU PHÂN LOẠI
# ==================================================

def input_category(field_name, display_name):
    """Nhập và kiểm tra dữ liệu phân loại."""

    valid_options = VALID_VALUES[field_name]

    while True:
        print(
            f"{display_name} "
            f"({', '.join(valid_options)})"
        )

        value = input("Nhập lựa chọn: ").strip()

        # Không phân biệt chữ hoa và chữ thường
        for option in valid_options:
            if value.lower() == option.lower():
                return option

        print("Giá trị không hợp lệ. Vui lòng nhập lại.\n")


# ==================================================
# 4. HÀM NHẬP SỐ NGUYÊN
# ==================================================

def input_integer(field_name, display_name, min_value, max_value):
    """Nhập số nguyên trong phạm vi cho phép."""

    while True:
        value = input(
            f"{display_name} ({min_value}-{max_value}): "
        ).strip()

        try:
            number = float(value)

            if not number.is_integer():
                print("Lỗi: Vui lòng nhập số nguyên.\n")
                continue

            number = int(number)

            if not min_value <= number <= max_value:
                print(
                    f"Lỗi: Giá trị phải từ "
                    f"{min_value} đến {max_value}.\n"
                )
                continue

            return number

        except ValueError:
            print("Lỗi: Vui lòng nhập một số hợp lệ.\n")


# ==================================================
# 5. NHẬP THÔNG TIN ĐƠN HÀNG
# ==================================================

def input_order():
    """Thu thập đầy đủ 10 đặc trưng của một đơn hàng."""

    print("\n========== NHẬP THÔNG TIN ĐƠN HÀNG ==========")

    order = {}

    order["Warehouse_block"] = input_category(
        "Warehouse_block", "Kho hàng"
    )

    order["Mode_of_Shipment"] = input_category(
        "Mode_of_Shipment", "Phương thức vận chuyển"
    )

    order["Customer_care_calls"] = input_integer(
        "Customer_care_calls",
        "Số lần gọi chăm sóc khách hàng",
        2, 7
    )

    order["Customer_rating"] = input_integer(
        "Customer_rating",
        "Đánh giá của khách hàng",
        1, 5
    )

    order["Cost_of_the_Product"] = input_integer(
        "Cost_of_the_Product",
        "Giá sản phẩm",
        96, 310
    )

    order["Prior_purchases"] = input_integer(
        "Prior_purchases",
        "Số lần mua hàng trước đây",
        2, 10
    )

    order["Product_importance"] = input_category(
        "Product_importance", "Mức độ quan trọng"
    )

    order["Gender"] = input_category(
        "Gender", "Giới tính"
    )

    order["Discount_offered"] = input_integer(
        "Discount_offered",
        "Mức giảm giá",
        1, 65
    )

    order["Weight_in_gms"] = input_integer(
        "Weight_in_gms",
        "Cân nặng sản phẩm (gram)",
        1001, 7846
    )

    return order


# ==================================================
# 6. DỰ ĐOÁN MỘT ĐƠN HÀNG
# ==================================================

def predict_delivery(order_data):
    """Trả về xác suất và kết quả phân loại."""

    # Đảm bảo tên và thứ tự cột giống dữ liệu huấn luyện
    columns = [
        "Warehouse_block",
        "Mode_of_Shipment",
        "Customer_care_calls",
        "Customer_rating",
        "Cost_of_the_Product",
        "Prior_purchases",
        "Product_importance",
        "Gender",
        "Discount_offered",
        "Weight_in_gms"
    ]

    input_df = pd.DataFrame(
        [order_data],
        columns=columns
    )

    # Xác suất thuộc lớp 1 = giao trễ
    late_probability = model.predict_proba(input_df)[0][LATE_INDEX]

    prediction = int(late_probability >= THRESHOLD)

    return {
        "late_probability": late_probability,
        "prediction": prediction,
        "status": (
            "CẢNH BÁO: Có nguy cơ giao trễ"
            if prediction == 1
            else "Dự đoán giao đúng hạn"
        )
    }


# ==================================================
# 7. HIỂN THỊ KẾT QUẢ
# ==================================================

def display_result(order_number, order, result):
    print("\n" + "=" * 50)
    print(f"KẾT QUẢ ĐƠN HÀNG #{order_number}")
    print("=" * 50)

    print("\nThông tin đơn hàng:")

    for key, value in order.items():
        print(f"- {key}: {value}")

    print(
        f"\nXác suất giao trễ: "
        f"{result['late_probability']:.2%}"
    )

    print(f"Ngưỡng phân loại: {THRESHOLD:.0%}")
    print(f"Kết quả: {result['status']}")


# ==================================================
# 8. CHƯƠNG TRÌNH CHÍNH: DỰ ĐOÁN NHIỀU ĐƠN
# ==================================================

def main():
    results = []
    order_number = 1

    print("=" * 50)
    print("HỆ THỐNG DỰ ĐOÁN RỦI RO GIAO HÀNG TRỄ")
    print("Mô hình: Random Forest")
    print(f"Ngưỡng phân loại: {THRESHOLD:.0%}")
    print("=" * 50)

    while True:
        order = input_order()

        result = predict_delivery(order)

        display_result(order_number, order, result)

        results.append({
            "Order": order_number,
            "Late_probability": result["late_probability"],
            "Prediction": result["prediction"],
            "Status": result["status"]
        })

        order_number += 1

        while True:
            choice = input(
                "\nBạn có muốn dự đoán đơn hàng tiếp theo? (c/k): "
            ).strip().lower()

            if choice in ["c", "k"]:
                break

            print("Vui lòng nhập c (có) hoặc k (không).")

        if choice == "k":
            break

    # Tổng kết các đơn hàng đã dự đoán
    print("\n" + "=" * 50)
    print("TỔNG KẾT DỰ ĐOÁN")
    print("=" * 50)

    summary = pd.DataFrame(results)

    print(summary.to_string(index=False))

    late_count = sum(
        item["Prediction"] == 1 for item in results
    )

    on_time_count = len(results) - late_count

    print(f"\nTổng số đơn hàng: {len(results)}")
    print(f"Số đơn có nguy cơ giao trễ: {late_count}")
    print(f"Số đơn dự đoán giao đúng hạn: {on_time_count}")

    # Lưu kết quả vào file CSV
    output_path = os.path.join(
        BASE_DIR, "models", "prediction_results.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nĐã lưu kết quả tại:\n{output_path}")


if __name__ == "__main__":
    main()
