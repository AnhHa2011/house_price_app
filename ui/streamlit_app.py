import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

# ------------------------------------------------------------------
# 1. SETUP & IMPORTS
# ------------------------------------------------------------------
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from engine.preprocess_pipeline import AmesPreprocessor
    from engine.predict_engine import HousePricePredictor
except ImportError:
    st.error("Lỗi Import: Không tìm thấy engine.")
    st.stop()

# ------------------------------------------------------------------
# 2. CONFIG & UTILS
# ------------------------------------------------------------------
st.set_page_config(page_title="Hệ thống Dự đoán Giá Nhà (AVM)", page_icon="🏡", layout="wide")


@st.cache_resource
def load_predictor():
    try:
        predictor = HousePricePredictor()
        if not predictor.is_ready: return None
        return predictor
    except:
        return None


PREDICTOR = load_predictor()


def get_index(options, value):
    try:
        val_str = str(value)
        if val_str in options: return options.index(val_str)
        return 0
    except:
        return 0


# --- HÀM CHUẨN HÓA DỮ LIỆU CHO RADAR CHART (MỚI) ---
def normalize_value(val, min_v, max_v):
    """Quy đổi giá trị bất kỳ về thang điểm 0-10"""
    if val is None: return 0
    try:
        val = float(val)
        if val >= max_v: return 10
        if val <= min_v: return 0
        return ((val - min_v) / (max_v - min_v)) * 10
    except:
        return 0


def get_radar_data(input_data):
    """Tạo dữ liệu cho biểu đồ Radar từ các yếu tố ảnh hưởng chính"""
    # Định nghĩa các trục và khoảng giá trị chuẩn (Min, Max) để chấm điểm
    # Ví dụ: Diện tích 3000sf được coi là 10 điểm
    axes = {
        'Chất lượng (Overall)': (input_data.get('OverallQual', 5), 1, 10),
        'Diện tích ở (Living)': (input_data.get('GrLivArea', 0), 500, 3000),
        'Diện tích đất (Lot)': (input_data.get('LotArea', 0), 2000, 20000),
        'Garage (Sức chứa)': (input_data.get('GarageCars', 0), 0, 4),
        'Tầng hầm (Size)': (input_data.get('TotalBsmtSF', 0), 0, 2000),
        'Độ mới (Năm xây)': (input_data.get('YearBuilt', 1900), 1950, 2010)
    }

    values = []
    labels = []
    for label, (val, min_v, max_v) in axes.items():
        score = normalize_value(val, min_v, max_v)
        values.append(score)
        labels.append(label)

    return labels, values


# TOP 20 FEATURES (Giữ nguyên cho phần giả lập)
TOP_20_FEATURES = {
    "Chất lượng Tổng thể": ("OverallQual", "cat_num", [1, 10]),
    "Diện tích ở (GrLivArea)": ("GrLivArea", "num", [500, 5000]),
    "Khu vực (Neighborhood)": ("Neighborhood", "cat", []),
    "Sức chứa Garage": ("GarageCars", "num", [0, 5]),
    "Diện tích Garage": ("GarageArea", "num", [0, 2000]),
    "Diện tích Hầm": ("TotalBsmtSF", "num", [0, 5000]),
    "Diện tích Tầng 1": ("1stFlrSF", "num", [0, 5000]),
    "Chất lượng Bếp": ("KitchenQual", "cat", ['Fa', 'TA', 'Gd', 'Ex']),
    "Phòng tắm Full": ("FullBath", "num", [0, 5]),
    "Năm xây dựng": ("YearBuilt", "num", [1800, 2025]),
    "Năm sửa chữa": ("YearRemodAdd", "num", [1950, 2025]),
    "Diện tích ốp gạch": ("MasVnrArea", "num", [0, 1000]),
    "Số lò sưởi": ("Fireplaces", "num", [0, 3]),
    "Diện tích hoàn thiện Hầm": ("BsmtFinSF1", "num", [0, 2000]),
    "Chất lượng Ngoại thất": ("ExterQual", "cat", ['Fa', 'TA', 'Gd', 'Ex']),
    "Chiều cao Hầm": ("BsmtQual", "cat", ['None', 'Fa', 'TA', 'Gd', 'Ex']),
    "Mặt tiền": ("LotFrontage", "num", [20, 200]),
    "Diện tích đất": ("LotArea", "num", [1000, 50000]),
    "Diện tích Hiên mở": ("OpenPorchSF", "num", [0, 500]),
    "Hoàn thiện Garage": ("GarageFinish", "cat", ['None', 'Unf', 'RFn', 'Fin'])
}


def get_complete_input_skeleton(predictor) -> Dict[str, Any]:
    cols = predictor.original_cols if predictor and predictor.is_ready else []
    feature_cols = [c for c in cols if c not in ['SalePrice', 'SalePrice_log']]
    default_dict = {col: np.nan for col in feature_cols}

    defaults = {
        'Id': 9999, 'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 60.0, 'LotArea': 10000,
        'Street': 'Pave', 'Alley': 'None', 'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub',
        'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'CollgCr', 'Condition1': 'Norm',
        'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '2Story', 'OverallQual': 7,
        'OverallCond': 5, 'YearBuilt': 2000, 'YearRemodAdd': 2002, 'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'BrkFace',
        'MasVnrArea': 0.0, 'ExterQual': 'TA', 'ExterCond': 'TA', 'Foundation': 'PConc',
        'BsmtQual': 'TA', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ',
        'BsmtFinSF1': 400, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 200,
        'TotalBsmtSF': 1000, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y',
        'Electrical': 'SBrkr', '1stFlrSF': 1000, '2ndFlrSF': 800, 'LowQualFinSF': 0,
        'GrLivArea': 1800, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 2,
        'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'TA',
        'TotRmsAbvGrd': 7, 'Functional': 'Typ', 'Fireplaces': 1, 'FireplaceQu': 'TA',
        'GarageType': 'Attchd', 'GarageYrBlt': 2000, 'GarageFinish': 'RFn',
        'GarageCars': 2, 'GarageArea': 500, 'GarageQual': 'TA', 'GarageCond': 'TA',
        'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 60, 'EnclosedPorch': 0,
        '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': 'None',
        'Fence': 'None', 'MiscFeature': 'None', 'MiscVal': 0, 'MoSold': 6,
        'YrSold': 2008, 'SaleType': 'WD', 'SaleCondition': 'Normal'
    }
    default_dict.update(defaults)
    return default_dict


# ==============================================================================
# FUNC 1: TAB DỰ ĐOÁN ĐƠN LẺ
# ==============================================================================
def render_single_prediction_tab(predictor, input_data):
    st.markdown("### Nhập thông tin chi tiết")
    st.info(" **Gợi ý:** Các trường diện tích không giới hạn giá trị tối đa. Hãy nhập con số thực tế.")

    with st.expander("📍 1. Vị trí & Lô đất (Location & Lot)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            opts_neigh = ['CollgCr', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr', 'MeadowV', 'IDOTRR', 'NAmes', 'Sawyer',
                          'OldTown', 'Edwards', 'Gilbert', 'SawyerW', 'Somerst', 'NWAmes', 'BrkSide', 'Crawfor',
                          'Mitchel', 'Timber', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste']
            input_data['Neighborhood'] = st.selectbox("Khu vực dân cư", opts_neigh,
                                                      index=get_index(opts_neigh, input_data.get('Neighborhood')))
            opts_zone = ['RL', 'RM', 'C (all)', 'FV', 'RH']
            input_data['MSZoning'] = st.selectbox("Phân loại quy hoạch", opts_zone,
                                                  index=get_index(opts_zone, input_data.get('MSZoning')))
            input_data['LotArea'] = st.number_input("Diện tích lô đất (sqft)", min_value=0,
                                                    value=int(input_data.get('LotArea', 10000)))
            input_data['Street'] = st.selectbox("Loại đường", ['Pave', 'Grvl'],
                                                index=get_index(['Pave', 'Grvl'], input_data.get('Street')))
        with c2:
            input_data['LotFrontage'] = st.number_input("Mặt tiền đường (ft)", min_value=0.0,
                                                        value=float(input_data.get('LotFrontage', 60.0)))
            opts_conf = ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3']
            input_data['LotConfig'] = st.selectbox("Cấu hình lô đất", opts_conf,
                                                   index=get_index(opts_conf, input_data.get('LotConfig')))
            opts_shape = ['Reg', 'IR1', 'IR2', 'IR3']
            input_data['LotShape'] = st.selectbox("Hình dáng lô đất", opts_shape,
                                                  index=get_index(opts_shape, input_data.get('LotShape')))
            input_data['Alley'] = st.selectbox("Lối vào hẻm", ['None', 'Grvl', 'Pave'],
                                               index=get_index(['None', 'Grvl', 'Pave'], input_data.get('Alley')))
        with c3:
            opts_cont = ['Lvl', 'Bnk', 'HLS', 'Low']
            input_data['LandContour'] = st.selectbox("Độ phẳng của đất", opts_cont,
                                                     index=get_index(opts_cont, input_data.get('LandContour')))
            input_data['LandSlope'] = st.selectbox("Độ dốc", ['Gtl', 'Mod', 'Sev'],
                                                   index=get_index(['Gtl', 'Mod', 'Sev'], input_data.get('LandSlope')))
            opts_cond = ['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe']
            input_data['Condition1'] = st.selectbox("Điều kiện ngoại cảnh 1", opts_cond,
                                                    index=get_index(opts_cond, input_data.get('Condition1')))
            input_data['Condition2'] = st.selectbox("Điều kiện ngoại cảnh 2", opts_cond,
                                                    index=get_index(opts_cond, input_data.get('Condition2')))

    with st.expander("🛠️ 2. Cấu trúc & Ngoại thất (Structure & Exterior)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data['MSSubClass'] = st.selectbox("Loại nhà (MSSubClass)",
                                                    [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180,
                                                     190], index=0)
            opts_bldg = ['1Fam', 'TwnhsE', 'Duplex', 'Twnhs', '2FmCon']
            input_data['BldgType'] = st.selectbox("Kiểu nhà", opts_bldg,
                                                  index=get_index(opts_bldg, input_data.get('BldgType')))
            opts_style = ['1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin']
            input_data['HouseStyle'] = st.selectbox("Phong cách nhà", opts_style,
                                                    index=get_index(opts_style, input_data.get('HouseStyle')))
            input_data['OverallQual'] = st.slider("Chất lượng tổng thể", 1, 10, int(input_data.get('OverallQual', 7)))
            input_data['OverallCond'] = st.slider("Tình trạng bảo quản", 1, 9, int(input_data.get('OverallCond', 5)))
        with c2:
            input_data['YearBuilt'] = st.number_input("Năm xây dựng", min_value=1800, max_value=2025,
                                                      value=int(input_data.get('YearBuilt', 2000)))
            input_data['YearRemodAdd'] = st.number_input("Năm sửa chữa", min_value=1800, max_value=2025,
                                                         value=int(input_data.get('YearRemodAdd', 2002)))
            input_data['RoofStyle'] = st.selectbox("Kiểu mái", ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'],
                                                   index=get_index(['Gable'], input_data.get('RoofStyle')))
            input_data['RoofMatl'] = st.selectbox("Vật liệu mái", ['CompShg', 'Tar&Grv', 'WdShngl', 'WdShake'], index=0)
            opts_ext = ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood',
                        'AsbShng', 'Stucco', 'BrkComm']
            input_data['Exterior1st'] = st.selectbox("Vật liệu ốp 1", opts_ext,
                                                     index=get_index(opts_ext, input_data.get('Exterior1st')))
            input_data['Exterior2nd'] = st.selectbox("Vật liệu ốp 2", opts_ext,
                                                     index=get_index(opts_ext, input_data.get('Exterior2nd')))
        with c3:
            input_data['MasVnrType'] = st.selectbox("Loại ốp gạch/đá", ['None', 'BrkFace', 'Stone', 'BrkCmn'],
                                                    index=get_index(['None', 'BrkFace'], input_data.get('MasVnrType')))
            input_data['MasVnrArea'] = st.number_input("Diện tích ốp (sqft)", min_value=0.0,
                                                       value=float(input_data.get('MasVnrArea', 0.0)))
            opts_qual = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
            input_data['ExterQual'] = st.selectbox("Chất lượng ngoại thất", opts_qual,
                                                   index=get_index(opts_qual, input_data.get('ExterQual')))
            input_data['ExterCond'] = st.selectbox("Tình trạng ngoại thất", opts_qual,
                                                   index=get_index(opts_qual, input_data.get('ExterCond')))
            input_data['Foundation'] = st.selectbox("Loại móng", ['PConc', 'CBlock', 'BrkTil', 'Wood'],
                                                    index=get_index(['PConc'], input_data.get('Foundation')))

    with st.expander("🛋️ 3. Không gian sống & Tiện ích (Living Space & Utilities)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data['1stFlrSF'] = st.number_input("Diện tích tầng 1", min_value=0,
                                                     value=int(input_data.get('1stFlrSF', 0)))
            input_data['2ndFlrSF'] = st.number_input("Diện tích tầng 2", min_value=0,
                                                     value=int(input_data.get('2ndFlrSF', 0)))
            input_data['LowQualFinSF'] = st.number_input("Diện tích thấp cấp", min_value=0,
                                                         value=int(input_data.get('LowQualFinSF', 0)))
            input_data['GrLivArea'] = st.number_input("Tổng diện tích ở (GrLivArea)", min_value=0,
                                                      value=int(input_data.get('GrLivArea', 1800)))
            input_data['FullBath'] = st.number_input("Số phòng tắm Full", min_value=0,
                                                     value=int(input_data.get('FullBath', 2)))
            input_data['HalfBath'] = st.number_input("Số phòng tắm Half", min_value=0,
                                                     value=int(input_data.get('HalfBath', 0)))
        with c2:
            input_data['BedroomAbvGr'] = st.number_input("Số phòng ngủ", min_value=0,
                                                         value=int(input_data.get('BedroomAbvGr', 3)))
            input_data['KitchenAbvGr'] = st.number_input("Số lượng bếp", min_value=0,
                                                         value=int(input_data.get('KitchenAbvGr', 1)))
            input_data['KitchenQual'] = st.selectbox("Chất lượng bếp", ['Ex', 'Gd', 'TA', 'Fa'],
                                                     index=get_index(['Ex', 'Gd', 'TA', 'Fa'],
                                                                     input_data.get('KitchenQual')))
            input_data['TotRmsAbvGrd'] = st.number_input("Tổng số phòng (trừ tắm)", min_value=0,
                                                         value=int(input_data.get('TotRmsAbvGrd', 7)))
            input_data['Functional'] = st.selectbox("Tính công năng",
                                                    ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
                                                    index=0)
            input_data['Utilities'] = st.selectbox("Tiện ích", ['AllPub', 'NoSewr', 'NoCsn', 'ELO'], index=0)
        with c3:
            input_data['Heating'] = st.selectbox("Hệ thống sưởi", ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
                                                 index=0)
            input_data['HeatingQC'] = st.selectbox("Chất lượng Sưởi", ['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=0)
            input_data['CentralAir'] = st.selectbox("Điều hòa trung tâm", ['Y', 'N'], index=0)
            input_data['Electrical'] = st.selectbox("Hệ thống điện", ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
                                                    index=0)
            input_data['Fireplaces'] = st.number_input("Số lò sưởi", min_value=0,
                                                       value=int(input_data.get('Fireplaces', 1)))
            input_data['FireplaceQu'] = st.selectbox("Chất lượng lò sưởi", ['None', 'Ex', 'Gd', 'TA', 'Fa', 'Po'],
                                                     index=get_index(['None', 'Ex', 'Gd', 'TA', 'Fa', 'Po'],
                                                                     input_data.get('FireplaceQu')))

    with st.expander("🚗 4. Tầng hầm & Garage (Basement & Garage)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            opts_bsmt = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None']
            input_data['BsmtQual'] = st.selectbox("Chiều cao Tầng hầm", opts_bsmt,
                                                  index=get_index(opts_bsmt, input_data.get('BsmtQual')))
            input_data['BsmtCond'] = st.selectbox("Tình trạng Tầng hầm", opts_bsmt,
                                                  index=get_index(opts_bsmt, input_data.get('BsmtCond')))
            input_data['BsmtExposure'] = st.selectbox("Độ thoáng hầm", ['Gd', 'Av', 'Mn', 'No', 'None'], index=3)
            input_data['BsmtFinType1'] = st.selectbox("Loại hoàn thiện 1",
                                                      ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None'], index=0)
            input_data['BsmtFinSF1'] = st.number_input("Diện tích hoàn thiện 1", min_value=0,
                                                       value=int(input_data.get('BsmtFinSF1', 0)))
            input_data['BsmtFinType2'] = st.selectbox("Loại hoàn thiện 2",
                                                      ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'None'], index=5)
            input_data['BsmtFinSF2'] = st.number_input("Diện tích hoàn thiện 2", min_value=0,
                                                       value=int(input_data.get('BsmtFinSF2', 0)))
        with c2:
            input_data['BsmtUnfSF'] = st.number_input("Diện tích chưa hoàn thiện", min_value=0,
                                                      value=int(input_data.get('BsmtUnfSF', 0)))
            input_data['TotalBsmtSF'] = st.number_input("Tổng diện tích hầm", min_value=0,
                                                        value=int(input_data.get('TotalBsmtSF', 0)))
            input_data['BsmtFullBath'] = st.number_input("Phòng tắm Full hầm", min_value=0,
                                                         value=int(input_data.get('BsmtFullBath', 0)))
            input_data['BsmtHalfBath'] = st.number_input("Phòng tắm Half hầm", min_value=0,
                                                         value=int(input_data.get('BsmtHalfBath', 0)))
            input_data['GarageType'] = st.selectbox("Vị trí Garage",
                                                    ['Attchd', 'Detchd', 'BuiltIn', 'Basment', 'CarPort', '2Types',
                                                     'None'], index=0)
            input_data['GarageYrBlt'] = st.number_input("Năm xây Garage", min_value=1900.0, max_value=2025.0,
                                                        value=float(input_data.get('GarageYrBlt', 2000.0)))
        with c3:
            input_data['GarageFinish'] = st.selectbox("Hoàn thiện Garage", ['Fin', 'RFn', 'Unf', 'None'], index=1)
            input_data['GarageCars'] = st.number_input("Sức chứa (xe)", min_value=0,
                                                       value=int(input_data.get('GarageCars', 2)))
            input_data['GarageArea'] = st.number_input("Diện tích Garage", min_value=0,
                                                       value=int(input_data.get('GarageArea', 500)))
            input_data['GarageQual'] = st.selectbox("Chất lượng Garage", opts_bsmt, index=2)
            input_data['GarageCond'] = st.selectbox("Tình trạng Garage", opts_bsmt, index=2)
            input_data['PavedDrive'] = st.selectbox("Đường lái xe", ['Y', 'P', 'N'], index=0)

    with st.expander("🌿 5. Tiện ích Ngoài trời & Giao dịch (Outdoor & Sale Info)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_data['WoodDeckSF'] = st.number_input("Sàn gỗ (sqft)", min_value=0,
                                                       value=int(input_data.get('WoodDeckSF', 0)))
            input_data['OpenPorchSF'] = st.number_input("Hiên mở (sqft)", min_value=0,
                                                        value=int(input_data.get('OpenPorchSF', 0)))
            input_data['EnclosedPorch'] = st.number_input("Hiên kín (sqft)", min_value=0,
                                                          value=int(input_data.get('EnclosedPorch', 0)))
            input_data['3SsnPorch'] = st.number_input("Hiên 3 mùa (sqft)", min_value=0,
                                                      value=int(input_data.get('3SsnPorch', 0)))
            input_data['ScreenPorch'] = st.number_input("Hiên lưới (sqft)", min_value=0,
                                                        value=int(input_data.get('ScreenPorch', 0)))
        with c2:
            input_data['PoolArea'] = st.number_input("Diện tích hồ bơi", min_value=0,
                                                     value=int(input_data.get('PoolArea', 0)))
            input_data['PoolQC'] = st.selectbox("Chất lượng hồ bơi", ['None', 'Ex', 'Gd', 'TA', 'Fa'], index=0)
            input_data['Fence'] = st.selectbox("Hàng rào", ['None', 'MnPrv', 'GdWo', 'MnWw', 'GdPrv'], index=0)
            input_data['MiscFeature'] = st.selectbox("Tính năng khác", ['None', 'Shed', 'Gar2', 'Othr', 'TenC'],
                                                     index=0)
            input_data['MiscVal'] = st.number_input("Giá trị tính năng khác", min_value=0,
                                                    value=int(input_data.get('MiscVal', 0)))
        with c3:
            input_data['MoSold'] = st.number_input("Tháng bán", min_value=1, max_value=12,
                                                   value=int(input_data.get('MoSold', 6)))
            input_data['YrSold'] = st.number_input("Năm bán", min_value=2006, max_value=2010,
                                                   value=int(input_data.get('YrSold', 2008)))
            input_data['SaleType'] = st.selectbox("Hình thức bán",
                                                  ['WD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth', 'CWD'],
                                                  index=0)
            input_data['SaleCondition'] = st.selectbox("Điều kiện bán",
                                                       ['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'],
                                                       index=0)

    # --- NÚT DỰ ĐOÁN ---
    st.markdown("---")

    with st.expander("Debug Dữ liệu (Dành cho Developer)"):
        st.write(input_data)

    confidence_level = st.selectbox(
        "Chọn độ tin cậy mong muốn:",
        [0.90, 0.95, 0.99],
        format_func=lambda x: f"{int(x * 100)}%",
        index=1
    )
    predict_btn = st.button("DỰ ĐOÁN GIÁ NGAY", type="primary", use_container_width=True)

    if predict_btn:
        try:
            with st.spinner("Đang tính toán..."):
                result = predictor.predict_single(input_data)
                price = result['predicted_price_usd']
                rmse = result['rmse_test']
                z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence_level]
                lower = price - (z * rmse)
                upper = price + (z * rmse)

                st.markdown(f"<h2 style='text-align: center; color: green;'>Giá dự đoán: ${price:,.0f}</h2>",
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.info(f"**Thấp nhất ({int(confidence_level * 100)}%):** ${lower:,.0f}")
                c2.info(f"**Cao nhất ({int(confidence_level * 100)}%):** ${upper:,.0f}")

        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")


# ==============================================================================
# FUNC 2: TAB DỰ ĐOÁN BATCH
# ==============================================================================
def render_batch_prediction_tab(predictor):
    st.header("Upload File CSV để Dự đoán Hàng loạt")
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write(f"**Dữ liệu đầu vào ({len(df_input)} dòng):**")
            st.dataframe(df_input.head())

            if st.button("Chạy Dự đoán"):
                with st.spinner(f"Đang xử lý..."):
                    df_output = predictor.predict_batch(df_input)
                    st.success("Hoàn tất!")
                    st.dataframe(df_output.head())
                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button("Tải xuống CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")


# ==============================================================================
# FUNC 3: TAB PHÂN TÍCH CHUYÊN SÂU (DASHBOARD ĐA CHIỀU)
# ==============================================================================
def render_analytics_tab(predictor, current_input_data):
    st.markdown("## Phân tích Chuyên sâu (Dashboard)")

    try:
        base_result = predictor.predict_single(current_input_data)
        base_price = base_result['predicted_price_usd']
        rmse = base_result['rmse_test']
    except:
        st.warning("Vui lòng nhập thông tin ở Tab 1 trước.")
        return

    st.success(f"**Định giá Cơ sở:** ${base_price:,.0f} | **Sai số (RMSE):** ${rmse:,.0f}")

    col_left, col_right = st.columns([1, 1])

    # --- CỘT TRÁI: BIỂU ĐỒ RADAR (Hồ sơ Sức mạnh) ---
    with col_left:
        st.subheader("1. Hồ sơ Sức mạnh Căn nhà")
        st.caption("So sánh các chỉ số quan trọng (quy đổi thang 10) để thấy điểm mạnh/yếu.")

        # Tạo dữ liệu radar từ hàm chuẩn hóa
        labels, values = get_radar_data(current_input_data)

        # Vẽ biểu đồ Radar
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name='Hiện tại',
            line_color='#1F77B4'
        ))

        # Cấu hình thang đo 0-10
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- CỘT PHẢI: GIẢ LẬP TOP 20 ---
    with col_right:
        st.subheader("2. Giả lập Chiến lược (Top 20)")

        feature_name = st.selectbox("Chọn yếu tố giả lập:", list(TOP_20_FEATURES.keys()))
        col_key, col_type, col_range = TOP_20_FEATURES[feature_name]

        current_val = current_input_data.get(col_key)
        # Fallback
        if current_val is None or pd.isna(current_val):
            current_val = col_range[0] if col_type != 'cat' else col_range[0]

        # Widget điều khiển
        new_val = None
        if col_type == 'cat':
            try:
                curr_idx = col_range.index(current_val)
            except:
                curr_idx = 0
            if col_key == 'Neighborhood':
                opts_neigh = ['CollgCr', 'Veenker', 'NoRidge', 'NridgHt', 'StoneBr', 'MeadowV', 'IDOTRR', 'NAmes',
                              'Sawyer', 'OldTown', 'Edwards', 'Gilbert', 'SawyerW', 'Somerst', 'NWAmes', 'BrkSide',
                              'Crawfor', 'Mitchel', 'Timber', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
                              'Blueste']
                new_val = st.selectbox("Chọn Khu vực mới:", opts_neigh, index=get_index(opts_neigh, current_val))
            else:
                new_val = st.select_slider(f"Mức độ {feature_name}:", options=col_range, value=col_range[curr_idx])
        elif col_type == 'cat_num':
            new_val = st.slider(f"Điểm {feature_name}:", min_value=col_range[0], max_value=col_range[1],
                                value=int(current_val))
        else:  # Numeric
            min_v, max_v = col_range
            new_val = st.slider(f"Giá trị {feature_name}:", min_value=min_v, max_value=max_v, value=int(current_val))

        sim_data = current_input_data.copy()
        sim_data[col_key] = new_val

        if st.button("Tính Tác Động", type="primary"):
            res = predictor.predict_single(sim_data)
            new_p = res['predicted_price_usd']
            delta = new_p - base_price
            st.metric("Giá trị Mới", f"${new_p:,.0f}", delta=f"${delta:,.0f}")

        # BIỂU ĐỒ ĐỘ NHẠY
        st.markdown("---")
        st.caption(f"Xu hướng giá theo {feature_name}:")

        analysis_data = []
        range_to_plot = []

        if col_type == 'cat' and col_key != "Neighborhood":
            range_to_plot = col_range
        elif col_type == 'cat_num':
            range_to_plot = list(range(col_range[0], col_range[1] + 1))
        elif col_type == 'num':
            step = (col_range[1] - col_range[0]) / 10
            range_to_plot = [int(col_range[0] + i * step) for i in range(11)]
        elif col_key == "Neighborhood":
            range_to_plot = list(set([current_val, 'NoRidge', 'NridgHt', 'StoneBr', 'OldTown', 'Edwards']))

        for val in range_to_plot:
            temp = current_input_data.copy()
            temp[col_key] = val
            try:
                r = predictor.predict_single(temp)
                analysis_data.append({'Value': str(val), 'Price': r['predicted_price_usd']})
            except:
                pass

        if analysis_data:
            df_chart = pd.DataFrame(analysis_data)
            if col_type in ['num', 'cat_num']:
                fig = px.line(df_chart, x='Value', y='Price', markers=True)
            else:
                fig = px.bar(df_chart, x='Value', y='Price', color='Price')

            # Highlight điểm hiện tại bằng Scatter riêng (Tránh lỗi add_vline)
            fig.add_trace(go.Scatter(
                x=[str(current_val)],
                y=[base_price],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name='Hiện tại'
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- 3. BÀI TOÁN ĐẦU TƯ ---
    st.markdown("---")
    st.subheader("3. Bài toán Đầu tư")
    with st.form("inv_form"):
        c1, c2, c3 = st.columns(3)
        with c1: rent = st.number_input("Giá thuê ($/tháng)", value=int(base_price * 0.007), step=50)
        with c2: down_pay = st.number_input("Trả trước (%)", 0, 100, 20)
        with c3: years = st.selectbox("Vay (năm)", [10, 15, 20, 30], index=2)

        if st.form_submit_button("Tính ROI"):
            loan = base_price * (1 - down_pay / 100)
            rate = 6.5 / 100 / 12
            n = years * 12
            monthly = loan * (rate * (1 + rate) ** n) / ((1 + rate) ** n - 1) if rate > 0 else loan / n
            cashflow = rent - monthly - 150
            cap = ((rent - 150) * 12) / base_price * 100

            k1, k2, k3 = st.columns(3)
            k1.metric("Trả góp/tháng", f"${monthly:,.0f}")
            k2.metric("Dòng tiền", f"${cashflow:,.0f}", delta=f"{cashflow:,.0f}")
            k3.metric("Cap Rate", f"{cap:.2f}%")


# ==============================================================================
# MAIN APP
# ==============================================================================
def run_streamlit_app():
    if not PREDICTOR: return

    input_data = get_complete_input_skeleton(PREDICTOR)

    tab1, tab2, tab3 = st.tabs([
        "Dự đoán Đơn lẻ",
        "Dự đoán Batch (CSV)",
        "Phân tích & Giả lập"
    ])

    with tab1: render_single_prediction_tab(PREDICTOR, input_data)
    with tab2: render_batch_prediction_tab(PREDICTOR)
    with tab3: render_analytics_tab(PREDICTOR, input_data)


if __name__ == '__main__':
    run_streamlit_app()