import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
from typing import Dict, Any, List, Union

# ------------------------------------------------------------------
# FIX LỖI IMPORT PATH: Đảm bảo Python thấy được package 'engine'
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------------------------------------

# 🚨 BẮT BUỘC: Import class AmesPreprocessor để giải mã file .pkl 🚨
from engine.preprocess_pipeline import AmesPreprocessor
# Import Engine
from engine.predict_engine import HousePricePredictor

# --- CẤU HÌNH CÁC HÀM HỖ TRỢ ---
try:
    PREDICTOR = HousePricePredictor()
except Exception:
    PREDICTOR = HousePricePredictor()  # Cố gắng tạo lần nữa để tránh lỗi fatal nếu chỉ là cảnh báo

# Danh sách cột tối thiểu (Fallback) nếu mô hình chưa sẵn sàng
MINIMAL_COLS = ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'OverallQual', 'GrLivArea',
                'YearBuilt', 'ExterQual', 'GarageCars', 'BsmtQual', 'KitchenQual', 'Neighborhood']


# Hàm tạo DataFrame Skeleton với các giá trị mặc định/NaN
def get_complete_input_skeleton(predictor: HousePricePredictor) -> Dict[str, Any]:
    cols = predictor.original_cols if predictor.is_ready else MINIMAL_COLS
    feature_cols = [c for c in cols if c not in ['SalePrice', 'SalePrice_log']]

    default_dict = {col: np.nan for col in feature_cols}

    # Điền các giá trị mặc định cho form
    default_dict.update({
        'Id': 1, 'MSSubClass': 20, 'MSZoning': 'RL', 'LotArea': 10000.0,
        'OverallQual': 7, 'YearBuilt': 2000, 'ExterQual': 'TA', 'GrLivArea': 1800.0,
        'GarageCars': 2, 'TotalBsmtSF': 1000, 'YearRemodAdd': 2002,
        'FullBath': 2, 'BsmtQual': 'TA', 'KitchenQual': 'TA', 'Neighborhood': 'CollgCr'
    })
    return default_dict


@st.cache_data
def process_batch_prediction(df_input: pd.DataFrame) -> pd.DataFrame:
    if not PREDICTOR.is_ready:
        raise RuntimeError("Mô hình không khả dụng.")

    df_result = PREDICTOR.predict_batch(df_input.copy())
    return df_result[['Id', 'SalePrice_Predicted']]


# --- UI CHÍNH ---
def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="Dự đoán Giá nhà (XGBoost)")
    st.title("🏡 Hệ thống Dự đoán Giá Nhà ở (XGBoost AVM)")

    # 🚨 KIỂM TRA TRẠNG THÁI MÔ HÌNH SẴN SÀNG 🚨
    if not PREDICTOR.is_ready:
        st.error(
            "🚨 Ứng dụng chưa sẵn sàng: Không tìm thấy các file mô hình (*.pkl). Vui lòng chạy **python train_and_export.py** để huấn luyện mô hình trước.")
        st.info(
            "Hãy đảm bảo các file `xgb_model.pkl`, `preprocess_pipeline.pkl`, và `metrics.pkl` nằm trong thư mục `models/`.")
        return

    tab1, tab2 = st.tabs(["Dự đoán Đơn lẻ (Form)", "Dự đoán Batch (CSV)"])

    # Lấy metrics để hiển thị
    rmse = PREDICTOR.RMSE_TEST_USD

    # ====================================================================
    # TAB 1: DỰ ĐOÁN ĐƠN LẺ (FORM)
    # ====================================================================
    with tab1:
        st.subheader("Dự đoán giá trị cho một căn nhà")
        input_data = get_complete_input_skeleton(PREDICTOR)
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data['OverallQual'] = st.slider("Chất lượng Tổng thể", 1, 10, input_data.get('OverallQual', 7))
            input_data['GrLivArea'] = st.number_input("Diện tích trên mặt đất (sqft)", min_value=300, max_value=5000,
                                                      value=int(input_data.get('GrLivArea', 1800)))
            input_data['ExterQual'] = st.selectbox("Chất lượng Ngoại thất", ['Ex', 'Gd', 'TA', 'Fa'],
                                                   index=['Ex', 'Gd', 'TA', 'Fa'].index(
                                                       input_data.get('ExterQual', 'TA')))
            input_data['KitchenQual'] = st.selectbox("Chất lượng Bếp", ['Ex', 'Gd', 'TA', 'Fa'],
                                                     index=['Ex', 'Gd', 'TA', 'Fa'].index(
                                                         input_data.get('KitchenQual', 'TA')))

        with col2:
            input_data['LotArea'] = st.number_input("Diện tích lô đất (sqft)", min_value=1000, max_value=50000,
                                                    value=int(input_data.get('LotArea', 10000)))
            input_data['TotalBsmtSF'] = st.number_input("Tổng diện tích tầng hầm (sqft)", min_value=0, max_value=4000,
                                                        value=1000)
            input_data['GarageCars'] = st.slider("Số chỗ đỗ xe trong Garage", 0, 4, input_data.get('GarageCars', 2))
            input_data['FullBath'] = st.slider("Số phòng tắm trên mặt đất", 0, 3, 2)

        with col3:
            input_data['YearBuilt'] = st.number_input("Năm Xây dựng", min_value=1800, max_value=2024,
                                                      value=int(input_data.get('YearBuilt', 2000)))
            input_data['YearRemodAdd'] = st.number_input("Năm Cải tạo cuối", min_value=1800, max_value=2024, value=2002)
            input_data['Neighborhood'] = st.selectbox("Khu vực Lân cận",
                                                      ['CollgCr', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr', 'MeadowV',
                                                       'IDOTRR', 'NAmes', 'Sawyer', 'OldTown'],
                                                      index=['CollgCr', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr',
                                                             'MeadowV', 'IDOTRR', 'NAmes', 'Sawyer', 'OldTown'].index(
                                                          input_data.get('Neighborhood', 'CollgCr')))
            input_data['BsmtQual'] = st.selectbox("Chất lượng Tầng hầm", ['Ex', 'Gd', 'TA', 'Fa', 'None'],
                                                  index=['Ex', 'Gd', 'TA', 'Fa', 'None'].index(
                                                      input_data.get('BsmtQual', 'TA')))

        if st.button("🚀 DỰ BÁO GIÁ", key='single_predict'):
            try:
                results = PREDICTOR.predict_single(input_data)

                st.success(f"Giá nhà dự đoán là: **${results['predicted_price_usd']:,.2f} USD**")
                st.info(
                    f"Khoảng tin cậy 95%: **${results['confidence_lower']:,.2f}** đến **${results['confidence_upper']:,.2f} USD** (Sai số RMSE: ${results['rmse_test']:,.2f})")

                with st.expander("Hiển thị Chi tiết Hiệu suất Mô hình"):
                    st.json(results['all_metrics'])
            except RuntimeError as e:
                st.error(f"Lỗi Dự đoán: {e}")

    # ====================================================================
    # TAB 2: DỰ ĐOÁN BATCH (CSV)
    # ====================================================================
    with tab2:
        st.subheader("Tải lên file CSV để dự đoán hàng loạt")
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.write("Dữ liệu đầu vào:")
            st.dataframe(df_input.head())

            if st.button("Bắt đầu Dự đoán Batch"):
                try:
                    with st.spinner("Đang xử lý và dự đoán..."):
                        df_output = process_batch_prediction(df_input)

                    st.success("✅ Dự đoán Batch hoàn tất!")
                    st.dataframe(df_output)

                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Tải về kết quả dự đoán (CSV)",
                        data=csv,
                        file_name='house_price_predictions.csv',
                        mime='text/csv',
                    )
                except RuntimeError as e:
                    st.error(f"Lỗi Batch: {e}")


if __name__ == '__main__':
    run_streamlit_app()