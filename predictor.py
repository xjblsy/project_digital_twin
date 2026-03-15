import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class FloodRiskPredictor:
    """洪水风险预测器"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.best_thresh = None
        self.is_loaded = False
        
    def load_model(self):
        """加载模型组件"""
        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'best_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'best_scaler.pkl'))
            
            with open(os.path.join(self.model_dir, 'best_feature_cols.txt'), 'r', encoding='utf-8') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            with open(os.path.join(self.model_dir, 'best_threshold.txt'), 'r', encoding='utf-8') as f:
                self.best_thresh = float(f.read().strip())
            
            self.is_loaded = True
            print(f"✅ 模型加载成功 | 特征数：{len(self.feature_cols)} | 阈值：{self.best_thresh}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            return False
    
    def generate_features(self, df_hist):
        """生成特征（与训练时完全一致）"""
        df = df_hist.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 1. 基础降雨累积
        if 'precipitation' in df.columns:
            df['rain_3d'] = df['precipitation'].shift(1).rolling(3, min_periods=1).sum()
            df['rain_7d'] = df['precipitation'].shift(1).rolling(7, min_periods=1).sum()
            df['rain_14d'] = df['precipitation'].shift(1).rolling(14, min_periods=1).sum()

        # 2. 暴雨日与水位急涨
        if 'precipitation' in df.columns:
            df['heavy_rain_day'] = (df['precipitation'] > 50).astype(int)
            df['heavy_rain_days_3d'] = df['heavy_rain_day'].shift(1).rolling(3, min_periods=1).sum()
            df['heavy_rain_days_7d'] = df['heavy_rain_day'].shift(1).rolling(7, min_periods=1).sum()

        df['rapid_rise'] = (df['water_level'].shift(1).diff() > 0.5).astype(int)
        df['rapid_rise_days_3d'] = df['rapid_rise'].shift(1).rolling(3, min_periods=1).sum()
        df['rapid_rise_days_7d'] = df['rapid_rise'].shift(1).rolling(7, min_periods=1).sum()

        # 3. 水位滑动统计
        for window in [3, 7, 14]:
            df[f'water_level_mean_{window}d'] = df['water_level'].shift(1).rolling(window, min_periods=1).mean()
            df[f'water_level_max_{window}d'] = df['water_level'].shift(1).rolling(window, min_periods=1).max()

        # 4. 滞后特征
        for lag in [1, 2, 3, 7]:
            df[f'water_level_lag_{lag}d'] = df['water_level'].shift(lag)
            if 'precipitation' in df.columns:
                df[f'precip_lag_{lag}d'] = df['precipitation'].shift(lag)

        # 5. 连续降雨日数
        if 'precipitation' in df.columns:
            rain_gt0 = (df['precipitation'].shift(1) > 0).astype(int)
            df['rain_days_3d'] = rain_gt0.rolling(3, min_periods=1).sum()
            df['rain_days_7d'] = rain_gt0.rolling(7, min_periods=1).sum()

        # 6. 前期降水指数（API）
        if 'precipitation' in df.columns:
            def compute_api(series, decay=0.9):
                if len(series) == 0 or pd.isna(series).all():
                    return np.nan
                weights = decay ** np.arange(len(series))[::-1]
                return np.nansum(series * weights)
            df['api_7d'] = df['precipitation'].shift(1).rolling(7, min_periods=1).apply(compute_api, raw=False)

        # 7. 时间特征
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_rainy_season'] = ((df['month'] >= 5) & (df['month'] <= 7)).astype(int)

        # 8. 物理交互特征
        if 'precipitation' in df.columns and 'water_level' in df.columns:
            df['precip_to_level_ratio'] = df['precipitation'] / (df['water_level'].shift(1) + 0.1)
            df['precip_to_level_ratio'] = df['precip_to_level_ratio'].clip(upper=10)

        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_index'] = df['temperature'] * (df['humidity'] / 100)

        if 'precipitation' in df.columns and 'humidity' in df.columns:
            df['wetness_index'] = (df['precipitation'] + 0.1) * (df['humidity'] / 100)

        # 9. 删除 NaN
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("数据不足以生成特征，请确保至少有 14 天的历史数据")
        
        return df.iloc[-1:][self.feature_cols]
    
    def predict(self, df_history):
        """执行预测"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        feature_row = self.generate_features(df_history)
        X_scaled = self.scaler.transform(feature_row)
        
        proba = self.model.predict_proba(X_scaled)[0]
        high_risk_prob = float(proba[2])
        y_pred = int(self.model.predict(X_scaled)[0])
        
        if high_risk_prob >= self.best_thresh:
            risk_level = 2
        else:
            risk_level = y_pred
        
        risk_map = {0: '低风险', 1: '中风险', 2: '高风险'}
        
        return {
            'risk': round((risk_level / 2), 2),
            'risk_level': risk_map[risk_level],
            'risk_level_code': risk_level,
            'high_risk_probability': round(high_risk_prob, 3),
            'probabilities': {
                'low': round(float(proba[0]), 3),
                'medium': round(float(proba[1]), 3),
                'high': round(high_risk_prob, 3)
            }
        }