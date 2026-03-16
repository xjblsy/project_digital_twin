from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import FloodRiskPredictor
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 允许跨域请求
CORS(app)

# 全局预测器实例
predictor = None

@app.before_first_request
def load_model_on_startup():
    """应用启动时自动加载模型"""
    global predictor
    try:
        # Railway 环境变量
        model_dir = os.environ.get('MODEL_DIR', './models')
        predictor = FloodRiskPredictor(model_dir)
        if predictor.load_model():
            logger.info("✅ 模型加载成功")
        else:
            logger.error("❌ 模型加载失败")
    except Exception as e:
        logger.error(f"模型加载异常：{e}")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_loaded if predictor else False,
        'timestamp': datetime.now().isoformat(),
        'service': 'Flood Risk Prediction API'
    })


@app.route('/predict', methods=['POST'])
def predict_risk():
    """
    风险预测接口
    
    请求体 (JSON):
    {
        "historical_data": [
            {"date": "2025-01-01", "water_level": 52.5, "precipitation": 10, "temperature": 25, "humidity": 70},
            ... 至少 14 天数据
        ]
    }
    
    返回:
    {
        "success": true,
        "prediction_date": "2025-01-16",
        "risk": 0.5,
        "risk_level": "中风险",
        "risk_level_code": 1,
        "high_risk_probability": 0.35
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'success': False,
                'error': '缺少 historical_data 字段'
            }), 400
        
        df_history = pd.DataFrame(data['historical_data'])
        
        # 验证必要列
        required_cols = ['date', 'water_level']
        missing_cols = [col for col in required_cols if col not in df_history.columns]
        if missing_cols:
            return jsonify({
                'success': False,
                'error': f'缺少必要列：{missing_cols}'
            }), 400
        
        # 验证数据量
        if len(df_history) < 14:
            return jsonify({
                'success': False,
                'error': f'历史数据不足，至少需要 14 天，当前只有 {len(df_history)} 天'
            }), 400
        
        # 执行预测
        result = predictor.predict(df_history)
        
        # 添加预测日期
        df_history['date'] = pd.to_datetime(df_history['date'])
        prediction_date = df_history['date'].max() + timedelta(days=1)
        result['prediction_date'] = prediction_date.strftime('%Y-%m-%d')
        result['success'] = True
        
        logger.info(f"预测成功 | 日期：{result['prediction_date']} | 风险：{result['risk_level']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测失败：{str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/', methods=['GET'])
def index():
    """首页"""
    return jsonify({
        'service': 'Flood Risk Prediction API',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': '健康检查',
            'POST /predict': '风险预测'
        },
        'docs': 'https://github.com/your-username/project_digital_twin'
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
