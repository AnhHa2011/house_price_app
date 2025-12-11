# --- CHÈN CODE NÀY VÀO PHẦN ĐẦU NOTEBOOK (SAU CÁC DÒNG IMPORT CHUNG) ---
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Union, Any

# Danh sách các cột KHÔNG PHẢI FEATURE (Target hoặc ID)
NON_FEATURE_COLS = ['SalePrice', 'SalePrice_log', 'Id']


class AmesPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.is_fitted = False
        self.train_columns = None
        self.original_input_columns = None

        # --- LƯU CÁC HẰNG SỐ CẦN THIẾT SAU KHI FIT ---
        self.global_imputation_values = {}
        self.neighborhood_medians = None
        self.low_variance_cols = []

    # ==========================================================================
    # PHƯƠNG THỨC XỬ LÝ DỮ LIỆU (Tích hợp các hàm con của bạn)
    # ==========================================================================

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_check = [col for col in df.columns if col != 'Id']

        # LƯU Ý: Nếu không tìm thấy trùng lặp, nó vẫn trả về DataFrame
        df = df.drop_duplicates(subset=cols_to_check)

        return df  # <<< Đảm bảo luôn trả về DataFrame

    def _handle_missing_values(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        print("\n>>> 2. XỬ LÝ GIÁ TRỊ THIẾU (MISSING VALUES)")

        # Nhóm 1: Location & Lot
        if 'Alley' in df.columns: df["Alley"] = df["Alley"].fillna("None")
        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
            if is_train:
                # FIT: Học các hằng số
                self.neighborhood_medians = df.groupby("Neighborhood")["LotFrontage"].median()
                df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
                self.global_imputation_values['LotFrontage'] = df["LotFrontage"].median()
            else:
                # TRANSFORM (FIX LỖI NONE-TYPE):
                if self.neighborhood_medians is not None:
                    # Nếu neighborhood_medians đã được load, sử dụng nó
                    global_median = self.global_imputation_values['LotFrontage']
                    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                        lambda x: x.fillna(self.neighborhood_medians.get(x.name, global_median))
                    )
                # Dòng này xử lý các giá trị NaN còn sót lại (bao gồm cả trường hợp self.neighborhood_medians là None)
                if 'LotFrontage' in self.global_imputation_values:
                    df["LotFrontage"] = df["LotFrontage"].fillna(self.global_imputation_values['LotFrontage'])

            # Nhóm 2: Mode/None/0 Imputation

        # Nhóm 2: Structure & Exterior
        mode_cols = ["Exterior1st", "Exterior2nd", "KitchenQual", "Functional", "Electrical", "SaleType",
                     "SaleCondition"]
        if 'MasVnrType' in df.columns: df["MasVnrType"] = df["MasVnrType"].fillna("None")
        if 'MasVnrArea' in df.columns: df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        for col in mode_cols:
            if col in df.columns:
                if is_train:
                    # FIT: Tính Mode và lưu vào self.global_imputation_values
                    self.global_imputation_values[col] = df[col].mode()[0]

                # TRANSFORM (FIX): Luôn sử dụng giá trị đã học
                fill_value = self.global_imputation_values.get(col)
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)  # Sử dụng giá trị đã học

        # Nhóm 3: Living & Utilities
        for col in ["FireplaceQu", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]:
            if col in df.columns: df[col] = df[col].fillna("None")

        # Nhóm 4: Basement & Garage
        bsmt_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
        for col in bsmt_cols:
            if col in df.columns: df[col] = df[col].fillna("None")

        garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        for col in garage_cols:
            if col in df.columns: df[col] = df[col].fillna("None")

        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            if is_train:
                self.global_imputation_values['GarageYrBlt'] = df["YearBuilt"].median()
            fill_value = self.global_imputation_values.get('GarageYrBlt', df['YearBuilt'].median())
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(fill_value)

        # Nhóm 5: Outdoor & Sale Info
        for col in ["PoolQC", "Fence", "MiscFeature"]:
            if col in df.columns: df[col] = df[col].fillna("None")
        if 'MiscVal' in df.columns: df["MiscVal"] = df["MiscVal"].fillna(0)
        for col in ["SaleType", "SaleCondition"]:
            if col in df.columns: df[col] = df[col].fillna(df[col].mode()[0])

        print("- Đã điền xong giá trị thiếu.")
        return df

    def _remove_low_variance(self, df: pd.DataFrame, is_train: bool, threshold=0.995) -> pd.DataFrame:
        print("\n>>> 3. LOẠI BỎ BIẾN ÍT THÔNG TIN (LOW VARIANCE)")
        low_variance_cols = []
        for col in df.columns:
            if not df[col].empty and df[col].nunique() > 0:
                top_val_freq = df[col].value_counts(normalize=True).values[0]
                if top_val_freq > threshold:
                    low_variance_cols.append(col)

        if low_variance_cols:
            print(f"- Phát hiện các cột ít thông tin: {low_variance_cols}")
            df = df.drop(columns=low_variance_cols)
            print("-> Đã loại bỏ các cột này.")
        else:
            print("- Không có cột nào quá đơn điệu.")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n>>> 4. XỬ LÝ NGOẠI LAI (OUTLIERS)")

        # 1. LotArea Capping
        if 'LotArea' in df.columns:
            count = (df['LotArea'] > 50000).sum()
            if count > 0:
                print(f"- Capping {count} lô đất > 50,000 sqft.")
                df.loc[df['LotArea'] > 50000, 'LotArea'] = 50000

        # 2. GrLivArea (Chỉ khi có SalePrice để check)
        if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
            outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index
            if len(outliers) > 0:
                print(f"- Loại bỏ {len(outliers)} điểm nhiễu GrLivArea lớn nhưng giá thấp.")
                df = df.drop(outliers)

        # 3. Logic YearRemodAdd < YearBuilt
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            mask = df["YearRemodAdd"] < df["YearBuilt"]
            if mask.sum() > 0:
                print(f"- Sửa {mask.sum()} lỗi YearRemodAdd < YearBuilt.")
                df.loc[mask, "YearRemodAdd"] = df.loc[mask, "YearBuilt"]

        return df

    def _fix_logic_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n>>> 5. SỬA LỖI LOGIC (INCONSISTENCY)")

        # Logic Kitchen
        if "KitchenQual" in df.columns and "KitchenAbvGr" in df.columns:
            mask = (df["KitchenAbvGr"] > 0) & (df["KitchenQual"] == "None")
            if mask.sum() > 0:
                print(f"- Sửa {mask.sum()} lỗi Kitchen có bếp nhưng thiếu Qual.")
                df.loc[mask, "KitchenQual"] = "TA"

        # Logic Fireplace
        if "Fireplaces" in df.columns and "FireplaceQu" in df.columns:
            mask = (df["Fireplaces"] > 0) & (df["FireplaceQu"] == "None")
            if mask.sum() > 0:
                print(f"- Sửa {mask.sum()} lỗi Fireplace có lò nhưng thiếu Qual.")
                df.loc[mask, "FireplaceQu"] = "TA"

        # Logic GarageYrBlt
        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            # Xử lý logic: Nếu Năm xây Garage > Năm bán -> Gán lại bằng Năm bán
            mask = df['GarageYrBlt'] > df['YrSold']
            if mask.sum() > 0:
                print(f"- Năm xây Garage > Năm bán -> Gán lại bằng Năm bán.")
                df.loc[mask, 'GarageYrBlt'] = df.loc[mask, 'YrSold']

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n>>> 6. KIẾN TẠO ĐẶC TRƯNG (FEATURE ENGINEERING)")

        # Tạo biến Boolean
        if 'Condition1' in df.columns:
            df['Is_Near_Park'] = df['Condition1'].isin(['PosN', 'PosA']).astype(int)
            df['Is_Near_Railroad'] = df['Condition1'].isin(['RRNn', 'RRAn', 'RRNe', 'RRAe']).astype(int)
        if 'LotConfig' in df.columns:
            df['Is_CulDSac'] = (df['LotConfig'] == 'CulDSac').astype(int)
        if 'LandContour' in df.columns:
            df['Is_Hillside'] = (df['LandContour'] == 'HLS').astype(int)

        # Neighborhood Rank & Interaction
        neigh_rank = {'NoRidge': 3, 'NridgHt': 3, 'StoneBr': 3, 'CollgCr': 2, 'Somerst': 2, 'MeadowV': 1, 'IDOTRR': 1}
        if 'Neighborhood' in df.columns and 'LotArea' in df.columns:
            df['Neigh_Rank_tmp'] = df['Neighborhood'].map(neigh_rank).fillna(1.5)
            df['Interaction_Area_Loc'] = df['LotArea'] * df['Neigh_Rank_tmp']
            df.drop(columns=['Neigh_Rank_tmp'], inplace=True)

        # House Age & Remodel
        if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
            df['IsNewHouse'] = (df['YearBuilt'] == df['YrSold']).astype(int)
            if 'YearRemodAdd' in df.columns:
                df['HasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
                df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
                df['YearsSinceBuilt'] = df['YrSold'] - df['YearBuilt']

        # Tổng diện tích
        if all(x in df.columns for x in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df['TotalLivingArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        if all(x in df.columns for x in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
            df['TotalBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5 + df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
            df['TotalBsmtBath'] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5

        # Diện tích trung bình phòng
        if 'GrLivArea' in df.columns and 'TotRmsAbvGrd' in df.columns:
            df['GrLivArea_per_Room'] = df.apply(
                lambda r: r['GrLivArea'] / r['TotRmsAbvGrd'] if r['TotRmsAbvGrd'] > 0 else 0, axis=1)

        # Log transform LotArea
        if 'LotArea' in df.columns:
            df['Log_LotArea'] = np.log1p(df['LotArea'])

        print("- Đã tạo xong các đặc trưng mới.")
        return df

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n>>> 7. MÃ HÓA DỮ LIỆU (ENCODING)")

        # Chuyển đổi kiểu dữ liệu
        cats = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
                'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']
        for col in cats:
            if col in df.columns: df[col] = df[col].astype('object')
        if 'MSSubClass' in df.columns: df['MSSubClass'] = df['MSSubClass'].astype(str)

        # 1. Ordinal Encoding (Map thủ công)
        mappings = {
            "LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0},
            "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0},
            "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
            "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
            "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
            "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
            "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "PavedDrive": {"Y": 2, "P": 1, "N": 0},
            "PoolQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "None": 0},
            "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0}
        }

        for col, mapping in mappings.items():
            if col in df.columns:
                # Chuẩn hóa về chuỗi trước khi map
                df[col] = df[col].astype(str).replace("nan", "None").replace("NA", "None")
                df[col] = df[col].map(mapping).fillna(0).astype(int)


        # 2. One-Hot Encoding
        # Lưu ý: Trong thực tế, nên dùng pd.get_dummies trên tập gộp (Train+Test) để tránh lệch cột
        # Ở đây giả sử làm trên df đơn lẻ
        nominal_cols = ['MSSubClass', 'BldgType', 'HouseStyle', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation']

        # Lọc ra các cột thực sự có trong DataFrame hiện tại
        cols_to_encode = [c for c in nominal_cols if c in df.columns]

        # Tạo prefix list chỉ cho các cột được encode
        prefixes_to_use = [c for c in nominal_cols if c in df.columns]

        if cols_to_encode:
            # Sử dụng các list đã lọc để đảm bảo độ dài khớp nhau
            df = pd.get_dummies(df, columns=cols_to_encode, prefix=prefixes_to_use)

        # 3. Label Encoding cho các cột Object còn lại
        remaining_objects = df.select_dtypes(include=['object']).columns
        for col in remaining_objects:
              df[col] = df[col].astype(str).astype("category").cat.codes

        # 3. Label Encoding cho các cột Object còn lại
        remaining_objects = df.select_dtypes(include=['object']).columns
        for col in remaining_objects:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        print(f"- Đã mã hóa xong. Kích thước hiện tại: {df.shape}")
        return df

    def _full_preprocess(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        df = df.copy()

        if not is_train and self.original_input_columns is not None:
            df = df.reindex(columns=self.original_input_columns, fill_value=np.nan)

        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df, is_train)

        df = self._remove_low_variance(df, is_train)

        if is_train:
            df = self._handle_outliers(df)

        df = self._fix_logic_errors(df)
        df = self._feature_engineering(df)
        df = self._encode_features(df)

        return df

    def fit(self, df: pd.DataFrame, y=None):
        self.original_input_columns = df.columns.tolist()
        X_processed = self._full_preprocess(df, is_train=True)
        feature_cols = [c for c in X_processed.columns if c not in NON_FEATURE_COLS]
        self.train_columns = feature_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy gọi .fit() trước.")

        X_processed = self._full_preprocess(df, is_train=False)

        X_final = pd.DataFrame(0, index=df.index, columns=self.train_columns)

        for col in X_processed.columns:
            if col in self.train_columns:
                X_final[col] = X_processed[col]

        return X_final[self.train_columns].values