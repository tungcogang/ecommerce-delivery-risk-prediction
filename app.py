
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==================================================
# 1. CẤU HÌNH TRANG
# ==================================================
st.set_page_config(
    page_title="Dự báo rủi ro giao hàng trễ",
    page_icon="🚚",
    layout="wide",
)

# ==================================================
# 2. ĐƯỜNG DẪN (tương đối theo thư mục gốc dự án)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LEGACY_HISTORY_PATH = os.path.join(MODELS_DIR, "prediction_results.csv")
APP_HISTORY_PATH = os.path.join(MODELS_DIR, "prediction_history_app.csv")
COMPARISON_PATH = os.path.join(MODELS_DIR, "final_model_comparison.csv")

# Màu sắc dùng chung cho biểu đồ
COLOR_BLUE = "#4DA3FF"
COLOR_RED = "#FF5C6C"
COLOR_TEXT = "#E8F0FF"

# Nhãn tiếng Việt (giá trị thật gửi vào mô hình vẫn là giá trị gốc)
LABEL_SHIPMENT = {"Ship": "Tàu thủy (Ship)", "Flight": "Máy bay (Flight)", "Road": "Đường bộ (Road)"}
LABEL_IMPORTANCE = {"low": "Thấp (low)", "medium": "Trung bình (medium)", "high": "Cao (high)"}
LABEL_GENDER = {"F": "Nữ (F)", "M": "Nam (M)"}


# ==================================================
# 3. HÀM NẠP DỮ LIỆU VÀ MÔ HÌNH (có cache để chạy nhanh)
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_predictor():
    # Dùng lại đúng mô hình, ngưỡng và hàm dự đoán trong src/predict_delivery.py
    from src import predict_delivery
    return predict_delivery


df = load_data()
predictor = load_predictor()
THRESHOLD = predictor.THRESHOLD
VALID = dict(predictor.VALID_VALUES)
# Danh sách kho lấy từ dữ liệu thật (A, B, C, D, F)
VALID["Warehouse_block"] = sorted(df["Warehouse_block"].unique())


# ==================================================
# 4. HÀM VẼ ĐỒNG HỒ ĐO XÁC SUẤT
# ==================================================
def make_gauge(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".2f", "font": {"color": COLOR_TEXT, "size": 48}},
            title={"text": "Xác suất giao trễ", "font": {"color": COLOR_TEXT, "size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR_TEXT, "tickfont": {"color": COLOR_TEXT}},
                "bar": {"color": COLOR_RED if probability >= THRESHOLD else COLOR_BLUE, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, THRESHOLD * 100], "color": "#14376B"},
                    {"range": [THRESHOLD * 100, 100], "color": "#5A2330"},
                ],
                "threshold": {
                    "line": {"color": "#FFD166", "width": 4},
                    "thickness": 0.85,
                    "value": THRESHOLD * 100,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_TEXT},
        height=340,
        margin=dict(l=30, r=30, t=70, b=10),
    )
    return fig



# ==================================================
# 4b. HÀM HỖ TRỢ BIỂU ĐỒ EDA
# ==================================================
TARGET = "Reached.on.Time_Y.N"
STATUS_MAP = {0: "Đúng hạn", 1: "Giao trễ"}
STATUS_COLORS = {"Đúng hạn": COLOR_BLUE, "Giao trễ": COLOR_RED}
FEATURE_ORDER = [
    "Warehouse_block", "Mode_of_Shipment", "Customer_care_calls",
    "Customer_rating", "Cost_of_the_Product", "Prior_purchases",
    "Product_importance", "Gender", "Discount_offered", "Weight_in_gms",
]

# ==================================================
# 5. THANH MENU BÊN TRÁI
# ==================================================

def style_fig(fig, height=360):
    """Nền trong suốt, chữ sáng, hợp với giao diện tối."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_TEXT},
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def late_rate_bar(data, column, title):
    """Biểu đồ cột: tỷ lệ giao trễ (%) theo từng giá trị của một cột."""
    grouped = (
        data.groupby(column)[TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "late_rate", "count": "orders"})
    )
    grouped["late_rate"] = grouped["late_rate"] * 100
    fig = px.bar(
        grouped,
        x=column,
        y="late_rate",
        text=grouped["late_rate"].map(lambda v: f"{v:.1f}%"),
        custom_data=["orders"],
        title=title,
        color_discrete_sequence=[COLOR_BLUE],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}<br>Tỷ lệ giao trễ: %{y:.2f}%<br>Số đơn: %{customdata[0]:,}<extra></extra>",
    )
    fig.update_yaxes(title="Tỷ lệ giao trễ (%)", range=[0, 100])
    fig.update_xaxes(title="")
    return style_fig(fig)
st.sidebar.title("🚚 Giao hàng thông minh")
st.sidebar.caption("Dự báo rủi ro giao hàng trễ trong thương mại điện tử")

page = st.sidebar.radio(
    "Chọn chức năng",
    [
        "📊 Tổng quan",
        "🔮 Dự đoán đơn hàng",
        "📁 Dự đoán hàng loạt",
        "🕘 Lịch sử dự đoán",
        "🧠 Thông tin mô hình",
    ],
)
# ==================================================
# 4c. HÀM DỰ ĐOÁN HÀNG LOẠT
# ==================================================
def predict_batch(data):
    """Dự đoán nhiều đơn hàng bằng đúng mô hình và ngưỡng trong predict_delivery.py."""
    features = data[FEATURE_ORDER].copy()
    probabilities = predictor.model.predict_proba(features)[:, predictor.LATE_INDEX]

    result = data.copy()
    result["Late_probability"] = probabilities
    result["Prediction"] = (probabilities >= THRESHOLD).astype(int)
    result["Status"] = result["Prediction"].map(
        {1: "CẢNH BÁO: Có nguy cơ giao trễ", 0: "Dự đoán giao đúng hạn"}
    )
    return result


def validate_batch(data):
    """Kiểm tra file tải lên. Trả về danh sách lỗi (rỗng nếu hợp lệ)."""
    errors = []

    missing = [c for c in FEATURE_ORDER if c not in data.columns]
    if missing:
        errors.append("Thiếu các cột: " + ", ".join(missing))
        return errors

    if data[FEATURE_ORDER].isnull().any().any():
        null_cols = data[FEATURE_ORDER].columns[data[FEATURE_ORDER].isnull().any()].tolist()
        errors.append("Có ô bị trống ở các cột: " + ", ".join(null_cols))

    for col in ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]:
        allowed = set(VALID[col])
        invalid = sorted(set(data[col].dropna().astype(str)) - allowed)
        if invalid:
            errors.append(f"Cột {col} có giá trị không hợp lệ: {invalid}. Chỉ chấp nhận: {sorted(allowed)}")

    numeric_cols = [
        "Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
        "Prior_purchases", "Discount_offered", "Weight_in_gms",
    ]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            errors.append(f"Cột {col} phải là số.")

    return errors

# ==================================================
# 4d. HÀM LỊCH SỬ DỰ ĐOÁN
# ==================================================
def save_to_history(order, probability, prediction):
    """Thêm một dòng vào file lịch sử của giao diện (không ghi đè dữ liệu cũ)."""
    row = {"Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    row.update({k: order[k] for k in FEATURE_ORDER})
    row["Late_probability"] = round(float(probability), 4)
    row["Prediction"] = int(prediction)
    row["Status"] = "CẢNH BÁO: Có nguy cơ giao trễ" if prediction == 1 else "Dự đoán giao đúng hạn"

    new_row = pd.DataFrame([row])
    if os.path.exists(APP_HISTORY_PATH):
        old = pd.read_csv(APP_HISTORY_PATH)
        new_row = pd.concat([old, new_row], ignore_index=True)
    new_row.to_csv(APP_HISTORY_PATH, index=False, encoding="utf-8-sig")


def read_history(path):
    """Đọc file lịch sử, trả về None nếu chưa có file."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def history_summary(hist):
    """Hiển thị 3 ô thống kê cho một bảng lịch sử."""
    n_total = len(hist)
    n_late = int((hist["Prediction"] == 1).sum())
    n_ok = n_total - n_late
    h1, h2, h3 = st.columns(3)
    h1.metric("Số lần dự đoán", f"{n_total:,}")
    h2.metric("Có nguy cơ giao trễ", f"{n_late:,}")
    h3.metric("Dự đoán đúng hạn", f"{n_ok:,}")


# ==================================================
# 6. NỘI DUNG TỪNG TRANG
# ==================================================
st.title("Phân tích hiệu suất vận chuyển và dự báo rủi ro giao hàng trễ")
st.subheader(page)

# ---------- TRANG 1: TỔNG QUAN ----------
if page == "📊 Tổng quan":
    total = len(df)
    late = int((df[TARGET] == 1).sum())
    on_time = total - late

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số đơn hàng", f"{total:,}")
    c2.metric("Giao trễ (nhãn 1)", f"{late:,}")
    c2.caption(f"Chiếm {late / total:.1%} tổng số đơn")
    c3.metric("Giao đúng hạn (nhãn 0)", f"{on_time:,}")
    c3.caption(f"Chiếm {on_time / total:.1%} tổng số đơn")

    plot_df = df.copy()
    plot_df["Trạng thái"] = plot_df[TARGET].map(STATUS_MAP)

    st.divider()

    # --- Hàng 1: tỷ lệ chung và phương thức vận chuyển ---
    r1a, r1b = st.columns(2)
    with r1a:
        pie = px.pie(
            plot_df,
            names="Trạng thái",
            color="Trạng thái",
            color_discrete_map=STATUS_COLORS,
            hole=0.5,
            title="Tỷ lệ giao đúng hạn và giao trễ",
        )
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(style_fig(pie), width="stretch")
    with r1b:
        st.plotly_chart(
            late_rate_bar(df, "Mode_of_Shipment", "Tỷ lệ giao trễ theo phương thức vận chuyển"),
            width="stretch",
        )

    # --- Hàng 2: kho hàng và mức độ quan trọng ---
    r2a, r2b = st.columns(2)
    with r2a:
        st.plotly_chart(
            late_rate_bar(df, "Warehouse_block", "Tỷ lệ giao trễ theo kho hàng"),
            width="stretch",
        )
    with r2b:
        st.plotly_chart(
            late_rate_bar(df, "Product_importance", "Tỷ lệ giao trễ theo mức độ quan trọng"),
            width="stretch",
        )

    # --- Hàng 3: giảm giá và cân nặng ---
    r3a, r3b = st.columns(2)
    with r3a:
        box1 = px.box(
            plot_df,
            x="Trạng thái",
            y="Discount_offered",
            color="Trạng thái",
            color_discrete_map=STATUS_COLORS,
            title="Mức giảm giá theo trạng thái giao hàng",
        )
        box1.update_xaxes(title="")
        box1.update_yaxes(title="Mức giảm giá")
        st.plotly_chart(style_fig(box1), width="stretch")
    with r3b:
        hist = px.histogram(
            plot_df,
            x="Weight_in_gms",
            color="Trạng thái",
            color_discrete_map=STATUS_COLORS,
            barmode="overlay",
            opacity=0.65,
            nbins=40,
            title="Phân phối cân nặng sản phẩm",
        )
        hist.update_xaxes(title="Cân nặng (gram)")
        hist.update_yaxes(title="Số đơn hàng")
        st.plotly_chart(style_fig(hist), width="stretch")

    with st.expander("Xem 10 dòng dữ liệu đầu tiên"):
        st.dataframe(df.head(10), hide_index=True, width="stretch")




# ---------- TRANG 2: DỰ ĐOÁN ĐƠN HÀNG ----------
elif page == "🔮 Dự đoán đơn hàng":
    st.write(
        f"Nhập thông tin đơn hàng rồi bấm **Dự đoán**. "
        f"Đơn hàng được xếp vào nhóm **giao trễ** khi xác suất giao trễ ≥ **{THRESHOLD:.0%}**."
    )

    with st.form("order_form"):
        col1, col2 = st.columns(2)

        with col1:
            warehouse = st.selectbox("Kho hàng", VALID["Warehouse_block"], index=0)
            shipment = st.selectbox(
                "Phương thức vận chuyển",
                VALID["Mode_of_Shipment"],
                index=VALID["Mode_of_Shipment"].index("Ship"),
                format_func=lambda x: LABEL_SHIPMENT[x],
            )
            importance = st.selectbox(
                "Mức độ quan trọng của sản phẩm",
                VALID["Product_importance"],
                index=VALID["Product_importance"].index("medium"),
                format_func=lambda x: LABEL_IMPORTANCE[x],
            )
            gender = st.selectbox(
                "Giới tính khách hàng",
                VALID["Gender"],
                index=0,
                format_func=lambda x: LABEL_GENDER[x],
            )
            weight = st.number_input("Cân nặng sản phẩm (gram)", min_value=1001, max_value=7846, value=3500, step=1)

        with col2:
            calls = st.number_input("Số lần gọi chăm sóc khách hàng", min_value=2, max_value=7, value=3, step=1)
            rating = st.number_input("Đánh giá của khách hàng (1-5)", min_value=1, max_value=5, value=4, step=1)
            cost = st.number_input("Giá sản phẩm", min_value=96, max_value=310, value=200, step=1)
            prior = st.number_input("Số lần mua hàng trước đây", min_value=2, max_value=10, value=3, step=1)
            discount = st.number_input("Mức giảm giá", min_value=1, max_value=65, value=10, step=1)

        submitted = st.form_submit_button("🔮 Dự đoán", type="primary")

    # --- Khi bấm Dự đoán: tính kết quả và LƯU VÀO session_state ---
    if submitted:
        order = {
            "Warehouse_block": warehouse,
            "Mode_of_Shipment": shipment,
            "Customer_care_calls": int(calls),
            "Customer_rating": int(rating),
            "Cost_of_the_Product": int(cost),
            "Prior_purchases": int(prior),
            "Product_importance": importance,
            "Gender": gender,
            "Discount_offered": int(discount),
            "Weight_in_gms": int(weight),
        }
        result = predictor.predict_delivery(order)
        st.session_state["last_prediction"] = {
            "order": order,
            "probability": float(result["late_probability"]),
            "prediction": int(result["prediction"]),
            "saved": False,
        }

    # --- Hiển thị kết quả từ session_state (không mất khi bấm nút khác) ---
    last = st.session_state.get("last_prediction")

    if last is not None:
        order = last["order"]
        probability = last["probability"]
        is_late = last["prediction"] == 1

        st.divider()
        left, right = st.columns([3, 2])

        with left:
            st.plotly_chart(make_gauge(probability), width="stretch")

        with right:
            st.metric("Xác suất giao trễ", f"{probability:.2%}")
            st.metric("Ngưỡng phân loại", f"{THRESHOLD:.0%}")
            if is_late:
                st.error("⚠️ CẢNH BÁO: Đơn hàng có nguy cơ giao trễ (nhãn 1)")
            else:
                st.success("✅ Đơn hàng được dự đoán giao đúng hạn (nhãn 0)")

        def save_last_prediction():
            item = st.session_state["last_prediction"]
            save_to_history(item["order"], item["probability"], item["prediction"])
            st.session_state["last_prediction"]["saved"] = True

        if last["saved"]:
            st.success("Đã lưu vào lịch sử (models/prediction_history_app.csv). Xem ở trang 🕘 Lịch sử dự đoán.")
        else:
            st.button("💾 Lưu vào lịch sử", on_click=save_last_prediction)

        with st.expander("Xem dữ liệu 10 đặc trưng đã gửi vào mô hình"):
            st.dataframe(pd.DataFrame([order])[FEATURE_ORDER], hide_index=True, width="stretch")

# ---------- CÁC TRANG CÒN LẠI (làm ở bước sau) ----------
elif page == "📁 Dự đoán hàng loạt":
    st.write(
        "Tải lên file CSV chứa thông tin nhiều đơn hàng. "
        f"Mỗi đơn được dự đoán bằng mô hình Random Forest với ngưỡng **{THRESHOLD:.0%}**."
    )

    with st.expander("File CSV cần có những cột nào?"):
        st.write("File cần đủ **10 cột** sau (thứ tự cột không quan trọng, cột thừa như ID sẽ được giữ lại):")
        st.code(", ".join(FEATURE_ORDER))

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Không đọc được file CSV: {exc}")
            batch_df = None

        if batch_df is not None:
            st.caption(f"Đã đọc {len(batch_df):,} dòng, {batch_df.shape[1]} cột.")
            problems = validate_batch(batch_df)

            if problems:
                for p in problems:
                    st.error(p)
            else:
                if st.button("🔮 Dự đoán hàng loạt", type="primary"):
                    st.session_state["batch_result"] = predict_batch(batch_df)

    if "batch_result" in st.session_state:
        res = st.session_state["batch_result"]
        n_total = len(res)
        n_late = int((res["Prediction"] == 1).sum())
        n_ok = n_total - n_late

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Tổng số đơn", f"{n_total:,}")
        m2.metric("Có nguy cơ giao trễ", f"{n_late:,}")
        m2.caption(f"Chiếm {n_late / n_total:.1%}")
        m3.metric("Dự đoán đúng hạn", f"{n_ok:,}")
        m3.caption(f"Chiếm {n_ok / n_total:.1%}")

        hist = px.histogram(
            res,
            x="Late_probability",
            nbins=20,
            title="Phân phối xác suất giao trễ",
            color_discrete_sequence=[COLOR_BLUE],
        )
        hist.add_vline(x=THRESHOLD, line_color="#FFD166", line_width=3,
                       annotation_text=f"Ngưỡng {THRESHOLD:.0%}", annotation_font_color="#FFD166")
        hist.update_xaxes(title="Xác suất giao trễ", range=[0, 1])
        hist.update_yaxes(title="Số đơn hàng")
        st.plotly_chart(style_fig(hist), width="stretch")

        show_df = res.copy()
        show_df["Late_probability"] = show_df["Late_probability"].round(4)
        st.dataframe(show_df, hide_index=True, width="stretch")

        csv_bytes = res.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải kết quả (CSV)",
            data=csv_bytes,
            file_name="ket_qua_du_doan_hang_loat.csv",
            mime="text/csv",
        )

elif page == "🕘 Lịch sử dự đoán":
    tab_app, tab_old = st.tabs(["Lịch sử từ giao diện", "Kết quả từ chương trình dòng lệnh"])

    with tab_app:
        hist_app = read_history(APP_HISTORY_PATH)
        if hist_app is None or hist_app.empty:
            st.info("Chưa có lịch sử. Hãy dự đoán ở trang Dự đoán đơn hàng rồi bấm 💾 Lưu vào lịch sử.")
        else:
            history_summary(hist_app)
            st.dataframe(hist_app.iloc[::-1], hide_index=True, width="stretch")
            st.download_button(
                "⬇️ Tải lịch sử (CSV)",
                data=hist_app.to_csv(index=False).encode("utf-8-sig"),
                file_name="lich_su_du_doan.csv",
                mime="text/csv",
            )

    with tab_old:
        hist_old = read_history(LEGACY_HISTORY_PATH)
        if hist_old is None or hist_old.empty:
            st.info("Chưa có file models/prediction_results.csv.")
        else:
            st.caption("Dữ liệu đọc trực tiếp từ models/prediction_results.csv (do src/predict_delivery.py tạo ra).")
            history_summary(hist_old)
            st.dataframe(hist_old, hide_index=True, width="stretch")

elif page == "🧠 Thông tin mô hình":
    if not os.path.exists(COMPARISON_PATH):
        st.error("Không tìm thấy models/final_model_comparison.csv")
    else:
        comp = pd.read_csv(COMPARISON_PATH)

        # Dòng của Random Forest ở đúng ngưỡng đang dùng
        row_df = comp[(comp["Model"] == "Random Forest") & ((comp["Threshold"] - THRESHOLD).abs() < 1e-9)]

        st.markdown(
            f"**Mô hình chính:** Random Forest (đã tinh chỉnh), file `models/random_forest_tuned.joblib`.  \n"
            f"**Ngưỡng phân loại:** {THRESHOLD} (xác suất giao trễ ≥ {THRESHOLD} thì xếp vào nhãn 1 - giao trễ).  \n"
            f"**Nhãn:** 0 = giao đúng hạn, 1 = giao trễ.  \n"
            f"**Số đặc trưng đầu vào:** {len(FEATURE_ORDER)}."
        )
        with st.expander("Danh sách 10 đặc trưng đầu vào"):
            st.code(", ".join(FEATURE_ORDER))

        st.divider()

        if row_df.empty:
            st.warning("File so sánh không có dòng Random Forest ở ngưỡng hiện tại.")
        else:
            r = row_df.iloc[0]
            st.subheader(f"Chỉ số đánh giá của Random Forest ở ngưỡng {THRESHOLD}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Accuracy", f"{r['Accuracy']:.2%}")
            k2.metric("Precision (giao trễ)", f"{r['Precision_late']:.2%}")
            k3.metric("Recall (giao trễ)", f"{r['Recall_late']:.2%}")
            k4.metric("F1 (giao trễ)", f"{r['F1_late']:.2%}")

            cm_col, note_col = st.columns([3, 2])
            with cm_col:
                tn, fp, fn, tp = int(r["TN"]), int(r["FP"]), int(r["FN"]), int(r["TP"])
                cm = go.Figure(
                    go.Heatmap(
                        z=[[tn, fp], [fn, tp]],
                        x=["Dự đoán đúng hạn", "Dự đoán giao trễ"],
                        y=["Thực tế đúng hạn", "Thực tế giao trễ"],
                        text=[[str(tn), str(fp)], [str(fn), str(tp)]],
                        texttemplate="%{text}",
                        textfont={"size": 22, "color": "#FFFFFF"},
                        colorscale=[[0, "#0F2347"], [1, "#4DA3FF"]],
                        showscale=False,
                    )
                )
                cm.update_layout(title="Ma trận nhầm lẫn")
                cm.update_yaxes(autorange="reversed")
                st.plotly_chart(style_fig(cm, height=380), width="stretch")
            with note_col:
                st.markdown(
                    f"- **TN = {tn}**: đúng hạn, dự đoán đúng hạn.\n"
                    f"- **FP = {fp}**: đúng hạn nhưng bị cảnh báo trễ.\n"
                    f"- **FN = {fn}**: thực tế trễ nhưng bỏ sót.\n"
                    f"- **TP = {tp}**: giao trễ, cảnh báo đúng.\n\n"
                    f"Ngưỡng thấp ({THRESHOLD}) giúp bắt được nhiều đơn trễ hơn (Recall cao), "
                    f"đổi lại có thêm cảnh báo nhầm (Precision thấp hơn)."
                )

        st.divider()
        st.subheader("So sánh Decision Tree và Random Forest")

        metric_cols = ["Accuracy", "Precision_late", "Recall_late", "F1_late"]
        metric_names = {
            "Accuracy": "Accuracy",
            "Precision_late": "Precision (giao trễ)",
            "Recall_late": "Recall (giao trễ)",
            "F1_late": "F1 (giao trễ)",
        }
        comp_plot = comp.copy()
        comp_plot["Mô hình / ngưỡng"] = comp_plot["Model"] + " (" + comp_plot["Threshold"].astype(str) + ")"
        long_df = comp_plot.melt(
            id_vars="Mô hình / ngưỡng", value_vars=metric_cols,
            var_name="Chỉ số", value_name="Giá trị",
        )
        long_df["Chỉ số"] = long_df["Chỉ số"].map(metric_names)

        main_label = f"Random Forest ({THRESHOLD})"
        color_map = {
            "Decision Tree (0.3)": "#3A5A8C",
            "Decision Tree (0.5)": "#6F8FBF",
            "Random Forest (0.5)": "#9CC4F5",
            main_label: "#4DA3FF",
        }
        # Mô hình chính viền sáng để nổi bật
        bar = px.bar(
            long_df, x="Chỉ số", y="Giá trị", color="Mô hình / ngưỡng",
            barmode="group", title="So sánh chỉ số các mô hình",
            color_discrete_map=color_map,
            text=long_df["Giá trị"].map(lambda v: f"{v:.2f}"),
        )
        bar.update_traces(textposition="outside")
        bar.update_traces(
            marker_line_color="#FFD166", marker_line_width=2.5,
            selector=dict(name=main_label),
        )
        bar.update_yaxes(range=[0, 1.05], title="Giá trị")
        bar.update_xaxes(title="")
        st.plotly_chart(style_fig(bar, height=440), width="stretch")
        st.caption(f"Viền vàng: mô hình chính đang dùng trong ứng dụng ({main_label}).")

        st.dataframe(
            comp.rename(columns={
                "Model": "Mô hình",
                "Threshold": "Ngưỡng",
                "Precision_late": "Precision (giao trễ)",
                "Recall_late": "Recall (giao trễ)",
                "F1_late": "F1 (giao trễ)",
            }),
            hide_index=True,
            width="stretch",
        )