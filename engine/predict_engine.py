import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union, Any, List
from .preprocess_pipeline import AmesPreprocessor
from xgboost import XGBRegressor


class HousePricePredictor:

    def __init__(self, model_path: str = 'models/xgb_model.pkl',
                 pipeline_path: str = 'models/preprocess_pipeline.pkl',
                 metrics_path: str = 'models/metrics.pkl'):

        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.metrics_path = metrics_path

        self.model = None
        self.pipeline = None
        self.metrics = {}
        self.RMSE_TEST_USD = 0.0
        self.original_cols = None
        self.is_ready = False

        # LƯU Ý: Lỗi xảy ra bên trong load_artifacts
        self.load_artifacts()

    def load_artifacts(self):
        try:
            self.model = joblib.load(self.model_path)
            self.pipeline = joblib.load(self.pipeline_path)
            self.metrics = joblib.load(self.metrics_path)
            print(f"self.metrics {self.metrics})")

            # --- SỬA LỖI KEY: SỬ DỤNG .get() THAY VÌ [] ---
            self.RMSE_TEST_USD = self.metrics['rmse']
            # -----------------------------------------------
            self.original_cols = self.pipeline.original_input_columns

            if self.original_cols is None:
                self.original_cols = self.metrics['ORIGINAL_COLS_WITH_TARGET']

            if self.original_cols is None:
                raise RuntimeError("Không tìm thấy danh sách cột gốc trong Pipeline hay Metrics.")

            self.is_ready = True
            print(f"INFO: Loaded Model, Pipeline, and RMSE: {self.RMSE_TEST_USD:,.2f} successfully.")

        except FileNotFoundError as e:
            self.is_ready = False
            print(f"WARNING: Model artifacts not found (Error: {e}). Vui lòng chạy train_and_export.py trước.")
        except Exception as e:
            # Bắt lỗi tổng quát hơn cho KeyError và các lỗi giải mã file .pkl khác
            self.is_ready = False
            print(f"ERROR: Failed to load artifacts or access keys. Lỗi: {e}")

    # --- CÁC PHƯƠNG THỨC KHÁC ĐƯỢC GIỮ NGUYÊN ---
    def _create_input_skeleton(self, input_dict: Dict[str, Union[str, float, int]]) -> pd.DataFrame:
        if self.original_cols is None:
            raise RuntimeError("Original columns list is missing.")

        original_features_cols = [col for col in self.original_cols if col not in ['SalePrice', 'SalePrice_log']]

        df_skeleton = pd.DataFrame(index=[0], columns=original_features_cols)

        for key, value in input_dict.items():
            if key in df_skeleton.columns:
                df_skeleton.loc[0, key] = value

        if 'Id' not in df_skeleton.columns or pd.isna(df_skeleton.loc[0, 'Id']):
            df_skeleton.loc[0, 'Id'] = 9999

        return df_skeleton.fillna(np.nan)

    def _post_process(self, y_log: np.ndarray) -> float:
        return np.expm1(y_log)[0]

    def predict_single(self, input_dict: Dict[str, Union[str, float, int]]) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Model artifacts are not loaded.")

        df_skeleton = self._create_input_skeleton(input_dict)
        X_processed = self.pipeline.transform(df_skeleton)
        y_log = self.model.predict(X_processed)
        predicted_price = self._post_process(y_log)
        ci_95 = 1.96 * self.RMSE_TEST_USD

        display_metrics = {k.replace('_USD', ''): round(v, 4) for k, v in self.metrics.items() if
                           isinstance(v, (float, int))}

        return {
            "predicted_price_usd": round(predicted_price, 2),
            "confidence_lower": round(predicted_price - ci_95, 2),
            "confidence_upper": round(predicted_price + ci_95, 2),
            "rmse_test": round(self.RMSE_TEST_USD, 2),
            "all_metrics": display_metrics
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready:
            raise RuntimeError("Model artifacts are not loaded.")

        df_reindexed = df.reindex(columns=self.original_cols, fill_value=np.nan)

        X_processed = self.pipeline.transform(df_reindexed)

        y_log = self.model.predict(X_processed)

        predicted_prices = np.expm1(y_log)

        if 'Id' in df.columns:
            df_result = df[['Id']].copy()
        else:
            df_result = pd.DataFrame(index=df.index)

        df_result['SalePrice_Predicted'] = predicted_prices

        return df_result