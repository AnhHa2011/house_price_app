import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from io import StringIO
from typing import Dict, Any, List, Union

# ------------------------------------------------------------------
# 1. FIX LỖI IMPORT PATH
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ------------------------------------------------------------------

# BẮT BUỘC: Import class AmesPreprocessor để giải mã file .pkl
from engine.preprocess_pipeline import AmesPreprocessor
# Import Engine
from engine.predict_engine import HousePricePredictor

# --- CẤU HÌNH CÁC HÀM HỖ TRỢ ---
try:
    PREDICTOR = HousePricePredictor()
except Exception:
    PREDICTOR = HousePricePredictor()

MINIMAL_COLS = ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'OverallQual', 'GrLivArea',
                'YearBuilt', 'ExterQual', 'GarageCars', 'BsmtQual', 'KitchenQual', 'Neighborhood']


# Hàm tạo DataFrame Skeleton với các giá trị mặc định/NaN
def get_complete_input_skeleton(predictor: HousePricePredictor) -> Dict[str, Any]:
    cols = predictor.original_cols if predictor.is_ready else MINIMAL_COLS
    feature_cols = [c for c in cols if c not in ['SalePrice', 'SalePrice_log']]

    default_dict = {col: np.nan for col in feature_cols}

    # Điền các giá trị mặc định (đã ép kiểu an toàn cho float/int)
    default_dict.update({
        'Id': 1, 'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 60.0, 'LotArea': 10000,
        'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub',
        'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'CollgCr', 'Condition1': 'Norm',
        'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '2Story', 'OverallQual': 7,
        'OverallCond': 5, 'YearBuilt': 2000, 'YearRemodAdd': 2002, 'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'BrkFace',
        'MasVnrArea': 100.0, 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
        'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ',
        'BsmtFinSF1': 700, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 300,
        'TotalBsmtSF': 1000, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
        'Electrical': 'SBrkr', '1stFlrSF': 800, '2ndFlrSF': 1000, 'LowQualFinSF': 0,
        'GrLivArea': 1800, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 2,
        'HalfBath': 0, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'TA',
        'TotRmsAbvGrd': 7, 'Functional': 'Typ', 'Fireplaces': 1, 'FireplaceQu': 'TA',
        'GarageType': 'Attchd', 'GarageYrBlt': 2000, 'GarageFinish': 'RFn',
        'GarageCars': 2, 'GarageArea': 480, 'GarageQual': 'TA', 'GarageCond': 'TA',
        'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 0, 'EnclosedPorch': 0,
        '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'None',
        'Fence': 'None', 'MiscFeature': 'None', 'MiscVal': 0, 'MoSold': 7,
        'YrSold': 2007, 'SaleType': 'WD', 'SaleCondition': 'Normal'
    })
    return default_dict


@st.cache_data
def process_batch_prediction(df_input: pd.DataFrame) -> pd.DataFrame:
    if not PREDICTOR.is_ready:
        raise RuntimeError("Mô hình không khả dụng.")

    df_result = PREDICTOR.predict_batch(df_input.copy())
    return df_result[['Id', 'SalePrice_Predicted']]


# Hàm hỗ trợ Selectbox (FIXED StreamlitAPIException)
def get_selectbox_index(options, default_value):
    try:
        default_value_str = str(default_value)
        if default_value_str in options:
            return options.index(default_value_str)
        return 0
    except Exception:
        return 0


# --- UI CHÍNH (ĐÃ MỞ RỘNG VÀ SỬA LỖI KIỂU DỮ LIỆU) ---
def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="Dự đoán Giá nhà (XGBoost)")
    st.title("🏡 Hệ thống Dự đoán Giá Nhà ở (XGBoost AVM)")

    if not PREDICTOR.is_ready:
        st.error(
            "🚨 Ứng dụng chưa sẵn sàng: Không tìm thấy các file mô hình (*.pkl). Vui lòng chạy **python train_and_export.py** để huấn luyện mô hình trước.")
        st.info(
            "Hãy đảm bảo các file `xgb_model.pkl`, `preprocess_pipeline.pkl`, và `metrics.pkl` nằm trong thư mục `models/`.")
        return

    tab1, tab2 = st.tabs(["Dự đoán Đơn lẻ (Form)", "Dự đoán Batch (CSV)"])

    with tab1:
        st.subheader("Dự đoán giá trị cho một căn nhà (79 Fields)")

        # --- Lấy dữ liệu Skeleton ---
        input_data = get_complete_input_skeleton(PREDICTOR)

        # ------------------------------------------------------------------
        # PHẦN 1: VỊ TRÍ & LÔ ĐẤT (12 Fields)
        # ------------------------------------------------------------------
        with st.expander("📍 1. Vị trí & Lô đất (Location & Lot)", expanded=True):
            col1_1, col1_2, col1_3 = st.columns(3)

            with col1_1:
                options_neigh = ['CollgCr', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr', 'MeadowV', 'IDOTRR', 'NAmes',
                                 'Sawyer', 'OldTown', 'SWISU']
                input_data['Neighborhood'] = st.selectbox("Khu vực dân cư", options_neigh,
                                                          index=get_selectbox_index(options_neigh,
                                                                                    input_data.get('Neighborhood')))

                options_zoning = ['RL', 'RM', 'C (all)', 'FV', 'RH']
                input_data['MSZoning'] = st.selectbox("Phân loại quy hoạch", options_zoning,
                                                      index=get_selectbox_index(options_zoning,
                                                                                input_data.get('MSZoning')))

                # FIXED: value must be int type
                value_lotarea = int(input_data.get('LotArea', 10000))
                input_data['LotArea'] = st.number_input("Diện tích lô đất (sqft)", min_value=1000, max_value=50000,
                                                        value=value_lotarea)

                options_street = ['Pave', 'Grvl']
                input_data['Street'] = st.selectbox("Loại đường tiếp cận", options_street,
                                                    index=get_selectbox_index(options_street, input_data.get('Street')))

            with col1_2:
                input_data['LotFrontage'] = st.number_input("Mặt tiền đường (ft)", min_value=0.0, max_value=200.0,
                                                            value=input_data.get('LotFrontage', 60.0))

                options_lotconf = ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
                input_data['LotConfig'] = st.selectbox("Cấu hình lô đất", options_lotconf,
                                                       index=get_selectbox_index(options_lotconf,
                                                                                 input_data.get('LotConfig')))

                options_lotshape = ['Reg', 'IR1', 'IR2', 'IR3']
                input_data['LotShape'] = st.selectbox("Hình dáng lô đất", options_lotshape,
                                                      index=get_selectbox_index(options_lotshape,
                                                                                input_data.get('LotShape')))

                options_landslope = ['Gtl', 'Mod', 'Sev']
                input_data['LandSlope'] = st.selectbox("Độ dốc", options_landslope,
                                                       index=get_selectbox_index(options_landslope,
                                                                                 input_data.get('LandSlope')))

            with col1_3:
                options_landcont = ['Lvl', 'Bnk', 'HLS', 'Low']
                input_data['LandContour'] = st.selectbox("Độ phẳng của đất", options_landcont,
                                                         index=get_selectbox_index(options_landcont,
                                                                                   input_data.get('LandContour')))

                options_alley = ['Grvl', 'Pave', 'None']
                input_data['Alley'] = st.selectbox("Lối vào hẻm", options_alley,
                                                   index=get_selectbox_index(options_alley, input_data.get('Alley')))

                options_cond1 = ['Norm', 'Feedr', 'PosN', 'RRAn', 'RRAe']
                input_data['Condition1'] = st.selectbox("Điều kiện ngoại cảnh (1)", options_cond1,
                                                        index=get_selectbox_index(options_cond1,
                                                                                  input_data.get('Condition1')))

                options_cond2 = ['Norm', 'Feedr', 'PosN', 'RRAn', 'RRAe']
                input_data['Condition2'] = st.selectbox("Điều kiện ngoại cảnh (2)", options_cond2,
                                                        index=get_selectbox_index(options_cond2,
                                                                                  input_data.get('Condition2')))

        # ------------------------------------------------------------------
        # PHẦN 2: CẤU TRÚC & NGOẠI THẤT (16 Fields)
        # ------------------------------------------------------------------
        with st.expander("🛠️ 2. Cấu trúc & Ngoại thất (Structure & Exterior)", expanded=False):
            col2_1, col2_2, col2_3 = st.columns(3)

            with col2_1:
                input_data['OverallQual'] = st.slider("Chất lượng vật liệu/hoàn thiện", 1, 10,
                                                      input_data.get('OverallQual', 7))
                input_data['OverallCond'] = st.slider("Tình trạng bảo quản", 1, 9, input_data.get('OverallCond', 5))
                input_data['YearBuilt'] = st.number_input("Năm xây dựng", min_value=1800, max_value=2024,
                                                          value=input_data.get('YearBuilt', 2000))
                input_data['YearRemodAdd'] = st.number_input("Năm sửa chữa/cải tạo", min_value=1800, max_value=2024,
                                                             value=input_data.get('YearRemodAdd', 2002))

                options_mstype = ['BrkFace', 'None', 'Stone', 'BrkCmn']
                input_data['MasVnrType'] = st.selectbox("Loại ốp gạch/đá trang trí", options_mstype,
                                                        index=get_selectbox_index(options_mstype,
                                                                                  input_data.get('MasVnrType')))

            with col2_2:
                options_msc = [20, 60, 50, 120, 30]  # MSSubClass top values
                input_data['MSSubClass'] = st.selectbox("Loại nhà (MSSubClass)", options_msc,
                                                        index=options_msc.index(input_data.get('MSSubClass', 20)))

                options_bldg = ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2FmCon']
                input_data['BldgType'] = st.selectbox("Kiểu nhà", options_bldg, index=get_selectbox_index(options_bldg,
                                                                                                          input_data.get(
                                                                                                              'BldgType')))

                options_hstyle = ['2Story', '1Story', '1.5Fin', 'SFoyer', 'SLvl']
                input_data['HouseStyle'] = st.selectbox("Phong cách nhà", options_hstyle,
                                                        index=get_selectbox_index(options_hstyle,
                                                                                  input_data.get('HouseStyle')))

                options_eq = ['Ex', 'Gd', 'TA', 'Fa']
                input_data['ExterQual'] = st.selectbox("Chất lượng ngoại thất", options_eq,
                                                       index=get_selectbox_index(options_eq,
                                                                                 input_data.get('ExterQual')))

                options_ec = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
                input_data['ExterCond'] = st.selectbox("Tình trạng ngoại thất", options_ec,
                                                       index=get_selectbox_index(options_ec,
                                                                                 input_data.get('ExterCond')))

            with col2_3:
                options_rstyle = ['Gable', 'Hip', 'Flat', 'Gambrel']
                input_data['RoofStyle'] = st.selectbox("Kiểu mái", options_rstyle,
                                                       index=get_selectbox_index(options_rstyle,
                                                                                 input_data.get('RoofStyle')))

                options_rmatl = ['CompShg', 'Tar&Grv', 'WdShngl']
                input_data['RoofMatl'] = st.selectbox("Vật liệu mái", options_rmatl,
                                                      index=get_selectbox_index(options_rmatl,
                                                                                input_data.get('RoofMatl', 'CompShg')))

                options_e1 = ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']
                input_data['Exterior1st'] = st.selectbox("Vật liệu ốp ngoài (1)", options_e1,
                                                         index=get_selectbox_index(options_e1,
                                                                                   input_data.get('Exterior1st')))
                input_data['Exterior2nd'] = st.selectbox("Vật liệu ốp ngoài (2)", options_e1,
                                                         index=get_selectbox_index(options_e1,
                                                                                   input_data.get('Exterior2nd')))

                input_data['MasVnrArea'] = st.number_input("Diện tích ốp gạch/đá (sqft)", min_value=0.0,
                                                           value=input_data.get('MasVnrArea', 100.0))

                options_found = ['PConc', 'CBlock', 'BrkTil', 'Wood']
                input_data['Foundation'] = st.selectbox("Loại móng nhà", options_found,
                                                        index=get_selectbox_index(options_found,
                                                                                  input_data.get('Foundation')))

        # ------------------------------------------------------------------
        # PHẦN 3: TẦNG HẦM & GARAGE (19 Fields)
        # ------------------------------------------------------------------
        with st.expander("🚗 3. Tầng hầm & Garage (Basement & Garage)", expanded=False):
            col3_1, col3_2, col3_3 = st.columns(3)

            with col3_1:  # Basement
                value_tbsf = int(input_data.get('TotalBsmtSF', 1000))
                input_data['TotalBsmtSF'] = st.number_input("Tổng diện tích tầng hầm (sqft)", min_value=0,
                                                            max_value=4000, value=value_tbsf)

                options_bq = ['Ex', 'Gd', 'TA', 'Fa', 'None']
                input_data['BsmtQual'] = st.selectbox("Chiều cao Tầng hầm (BsmtQual)", options_bq,
                                                      index=get_selectbox_index(options_bq, input_data.get('BsmtQual')))

                options_bc = ['Gd', 'TA', 'Fa', 'None']
                input_data['BsmtCond'] = st.selectbox("Tình trạng tầng hầm (BsmtCond)", options_bc,
                                                      index=get_selectbox_index(options_bc, input_data.get('BsmtCond')))

                options_be = ['Gd', 'Av', 'Mn', 'No', 'None']
                input_data['BsmtExposure'] = st.selectbox("Độ thoáng tầng hầm", options_be,
                                                          index=get_selectbox_index(options_be,
                                                                                    input_data.get('BsmtExposure',
                                                                                                   'No')))

                options_bft1 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None']
                input_data['BsmtFinType1'] = st.selectbox("Loại hoàn thiện 1", options_bft1,
                                                          index=get_selectbox_index(options_bft1,
                                                                                    input_data.get('BsmtFinType1')))

                value_bsf1 = int(input_data.get('BsmtFinSF1', 700))
                input_data['BsmtFinSF1'] = st.number_input("Diện tích hoàn thiện 1 (sqft)", min_value=0, max_value=3000,
                                                           value=value_bsf1)

            with col3_2:  # Bsmt Finishing & Garage Info
                options_bft2 = ['Unf', 'Rec', 'LwQ', 'None']
                input_data['BsmtFinType2'] = st.selectbox("Loại hoàn thiện 2", options_bft2,
                                                          index=get_selectbox_index(options_bft2,
                                                                                    input_data.get('BsmtFinType2')))

                value_bsf2 = int(input_data.get('BsmtFinSF2', 0))
                input_data['BsmtFinSF2'] = st.number_input("Diện tích hoàn thiện 2 (sqft)", min_value=0, max_value=2000,
                                                           value=value_bsf2)

                value_unf = int(input_data.get('BsmtUnfSF', 300))
                input_data['BsmtUnfSF'] = st.number_input("Diện tích chưa hoàn thiện (sqft)", min_value=0,
                                                          max_value=3000, value=value_unf)

                options_gt = ['Attchd', 'Detchd', 'BuiltIn', 'None']
                input_data['GarageType'] = st.selectbox("Vị trí garage", options_gt,
                                                        index=get_selectbox_index(options_gt,
                                                                                  input_data.get('GarageType')))

                options_gf = ['Fin', 'RFn', 'Unf', 'None']
                input_data['GarageFinish'] = st.selectbox("Mức độ hoàn thiện bên trong", options_gf,
                                                          index=get_selectbox_index(options_gf,
                                                                                    input_data.get('GarageFinish')))

                input_data['GarageCars'] = st.slider("Sức chứa (số xe)", 0, 4, input_data.get('GarageCars', 2))

            with col3_3:  # Garage
                value_garea = int(input_data.get('GarageArea', 480))
                input_data['GarageArea'] = st.number_input("Diện tích garage (sqft)", min_value=0, max_value=1200,
                                                           value=value_garea)

                input_data['GarageYrBlt'] = st.number_input("Năm xây garage", min_value=1900, max_value=2024,
                                                            value=input_data.get('GarageYrBlt', 2000))

                options_gq = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
                input_data['GarageQual'] = st.selectbox("Chất lượng garage", options_gq,
                                                        index=get_selectbox_index(options_gq,
                                                                                  input_data.get('GarageQual', 'TA')))
                input_data['GarageCond'] = st.selectbox("Tình trạng garage", options_gq,
                                                        index=get_selectbox_index(options_gq,
                                                                                  input_data.get('GarageCond', 'TA')))

                input_data['BsmtFullBath'] = st.slider("Phòng tắm Full dưới hầm", 0, 3,
                                                       input_data.get('BsmtFullBath', 1))
                input_data['BsmtHalfBath'] = st.slider("Phòng tắm Half dưới hầm", 0, 2,
                                                       input_data.get('BsmtHalfBath', 0))

                options_paved = ['Y', 'P', 'N']
                input_data['PavedDrive'] = st.selectbox("Đường lái xe vào", options_paved,
                                                        index=get_selectbox_index(options_paved,
                                                                                  input_data.get('PavedDrive', 'Y')))

        # ------------------------------------------------------------------
        # PHẦN 4: KHÔNG GIAN SỐNG & TIỆN ÍCH (18 Fields)
        # ------------------------------------------------------------------
        with st.expander("🛋️ 4. Không gian sống & Tiện ích (Living Space & Utilities)", expanded=False):
            col4_1, col4_2, col4_3 = st.columns(3)

            with col4_1:  # Living Space
                # FIXED: value must be int type
                value_grliv = int(input_data.get('GrLivArea', 1800))
                input_data['GrLivArea'] = st.number_input("Tổng diện tích ở trên mặt đất (sqft)", min_value=500,
                                                          max_value=5000, value=value_grliv)

                value_1sf = int(input_data.get('1stFlrSF', 800))
                input_data['1stFlrSF'] = st.number_input("Diện tích tầng 1 (sqft)", min_value=500, max_value=3000,
                                                         value=value_1sf)

                value_2sf = int(input_data.get('2ndFlrSF', 1000))
                input_data['2ndFlrSF'] = st.number_input("Diện tích tầng 2 (sqft)", min_value=0, max_value=2000,
                                                         value=value_2sf)

                value_lqfs = int(input_data.get('LowQualFinSF', 0))
                input_data['LowQualFinSF'] = st.number_input("Diện tích hoàn thiện chất lượng thấp (sqft)", min_value=0,
                                                             max_value=1000, value=value_lqfs)

            with col4_2:  # Bath & Kitchen
                input_data['FullBath'] = st.slider("Số phòng tắm Full (trên đất)", 0, 3, input_data.get('FullBath', 2))
                input_data['HalfBath'] = st.slider("Số phòng tắm Half (trên đất)", 0, 2, input_data.get('HalfBath', 0))
                input_data['BedroomAbvGr'] = st.slider("Số phòng ngủ", 0, 8, input_data.get('BedroomAbvGr', 3))
                input_data['KitchenAbvGr'] = st.slider("Số lượng bếp", 0, 3, input_data.get('KitchenAbvGr', 1))

                options_kq = ['Ex', 'Gd', 'TA', 'Fa']
                input_data['KitchenQual'] = st.selectbox("Chất lượng bếp", options_kq,
                                                         index=get_selectbox_index(options_kq,
                                                                                   input_data.get('KitchenQual')))

                input_data['TotRmsAbvGrd'] = st.slider("Tổng số phòng (trừ tắm)", 3, 12,
                                                       input_data.get('TotRmsAbvGrd', 7))

            with col4_3:  # Utility & Fireplace
                options_func = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
                input_data['Functional'] = st.selectbox("Tính công năng của nhà", options_func,
                                                        index=get_selectbox_index(options_func,
                                                                                  input_data.get('Functional', 'Typ')))

                options_util = ['AllPub', 'NoSewr', 'NoCsn', 'ELO']
                input_data['Utilities'] = st.selectbox("Tiện ích (Điện, nước, gas)", options_util,
                                                       index=get_selectbox_index(options_util,
                                                                                 input_data.get('Utilities', 'AllPub')))

                options_heat = ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor']
                input_data['Heating'] = st.selectbox("Hệ thống sưởi", options_heat,
                                                     index=get_selectbox_index(options_heat,
                                                                               input_data.get('Heating', 'GasA')))

                options_hqc = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
                input_data['HeatingQC'] = st.selectbox("Chất lượng sưởi", options_hqc,
                                                       index=get_selectbox_index(options_hqc,
                                                                                 input_data.get('HeatingQC', 'Ex')))

                options_ca = ['Y', 'N']
                input_data['CentralAir'] = st.selectbox("Điều hòa trung tâm", options_ca,
                                                        index=get_selectbox_index(options_ca,
                                                                                  input_data.get('CentralAir', 'Y')))

                options_elec = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
                input_data['Electrical'] = st.selectbox("Hệ thống điện", options_elec,
                                                        index=get_selectbox_index(options_elec,
                                                                                  input_data.get('Electrical',
                                                                                                 'SBrkr')))

                input_data['Fireplaces'] = st.slider("Số lò sưởi", 0, 3, input_data.get('Fireplaces', 1))

                options_fq = ['Gd', 'TA', 'Ex', 'None']
                input_data['FireplaceQu'] = st.selectbox("Chất lượng Lò sưởi", options_fq,
                                                         index=get_selectbox_index(options_fq,
                                                                                   input_data.get('FireplaceQu', 'TA')))

        # ------------------------------------------------------------------
        # PHẦN 5: NGOÀI TRỜI & GIAO DỊCH (14 Fields)
        # ------------------------------------------------------------------
        with st.expander("🌿 5. Ngoài trời & Thông tin giao dịch (Outdoor & Sale Info)", expanded=False):
            col5_1, col5_2, col5_3 = st.columns(3)

            with col5_1:  # Porches & Decks
                value_wdsf = int(input_data.get('WoodDeckSF', 0))
                input_data['WoodDeckSF'] = st.number_input("Diện tích sàn gỗ ngoài trời (sqft)", min_value=0,
                                                           max_value=1000, value=value_wdsf)

                value_opsf = int(input_data.get('OpenPorchSF', 0))
                input_data['OpenPorchSF'] = st.number_input("Diện tích hiên mở (sqft)", min_value=0, max_value=600,
                                                            value=value_opsf)

                value_epsf = int(input_data.get('EnclosedPorch', 0))
                input_data['EnclosedPorch'] = st.number_input("Diện tích hiên kín (sqft)", min_value=0, max_value=400,
                                                              value=value_epsf)

                value_3spsf = int(input_data.get('3SsnPorch', 0))
                input_data['3SsnPorch'] = st.number_input("Hiên 3 mùa (sqft)", min_value=0, max_value=400,
                                                          value=value_3spsf)

                value_scrsf = int(input_data.get('ScreenPorch', 0))
                input_data['ScreenPorch'] = st.number_input("Hiên có lưới che (sqft)", min_value=0, max_value=400,
                                                            value=value_scrsf)

            with col5_2:  # Pool & Fence
                value_poolarea = int(input_data.get('PoolArea', 0))
                input_data['PoolArea'] = st.number_input("Diện tích Hồ bơi (sqft)", min_value=0, max_value=800,
                                                         value=value_poolarea)

                options_pqc = ['Ex', 'Gd', 'TA', 'Fa', 'None']
                input_data['PoolQC'] = st.selectbox("Chất lượng Hồ bơi", options_pqc,
                                                    index=get_selectbox_index(options_pqc,
                                                                              input_data.get('PoolQC', 'None')))

                options_fence = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'None']
                input_data['Fence'] = st.selectbox("Hàng rào", options_fence, index=get_selectbox_index(options_fence,
                                                                                                        input_data.get(
                                                                                                            'Fence',
                                                                                                            'None')))

                options_miscf = ['Shed', 'Gar2', 'Othr', 'None']
                input_data['MiscFeature'] = st.selectbox("Các tính năng khác", options_miscf,
                                                         index=get_selectbox_index(options_miscf,
                                                                                   input_data.get('MiscFeature',
                                                                                                  'None')))

                input_data['MiscVal'] = st.number_input("Giá trị tính năng khác ($)", min_value=0, max_value=10000,
                                                        value=input_data.get('MiscVal', 0))

            with col5_3:  # Sale Info
                input_data['MoSold'] = st.slider("Tháng bán", 1, 12, input_data.get('MoSold', 7))
                input_data['YrSold'] = st.number_input("Năm bán", min_value=2006, max_value=2010,
                                                       value=input_data.get('YrSold', 2007))

                options_st = ['WD', 'New', 'COD', 'Con']
                input_data['SaleType'] = st.selectbox("Hình thức bán", options_st,
                                                      index=get_selectbox_index(options_st, input_data.get('SaleType')))

                options_sc = ['Normal', 'Partial', 'Abnorml', 'Family', 'Alloca', 'AdjLand']
                input_data['SaleCondition'] = st.selectbox("Điều kiện bán", options_sc,
                                                           index=get_selectbox_index(options_sc,
                                                                                     input_data.get('SaleCondition')))

        # ------------------------------------------------------------------
        # PHẦN DỰ ĐOÁN
        # ------------------------------------------------------------------
        st.markdown("---")
        if st.button(" DỰ ĐOÁN GIÁ NHÀ", key='single_predict'):
            try:
                results = PREDICTOR.predict_single(input_data)

                # Hiển thị kết quả
                st.success(f"Giá nhà dự đoán là: **${results['predicted_price_usd']:,.2f} USD**")
                st.info(
                    f"Khoảng tin cậy 95%: **${results['confidence_lower']:,.2f}** đến **${results['confidence_upper']:,.2f} USD** (Sai số RMSE: ${results['rmse_test']:,.2f})")

                with st.expander("Hiển thị Chi tiết Hiệu suất Mô hình"):
                    st.json(results['all_metrics'])
            except RuntimeError as e:
                st.error(f"Lỗi Dự đoán: {e}")
            except Exception as e:
                st.error(f"Lỗi không xác định trong quá trình dự đoán: {e}")

    # --- TAB 2 (Dự đoán Batch) --- (Giữ nguyên)
    with tab2:
        st.subheader("Tải lên file CSV để dự đoán hàng loạt")
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.write("Dữ liệu đầu vào:")
            st.dataframe(df_input.head())

            if st.button("Bắt đầu Dự đoán"):
                try:
                    with st.spinner("Đang xử lý và dự đoán..."):
                        df_output = process_batch_prediction(df_input)

                    st.success(" Dự đoán hoàn tất!")
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
