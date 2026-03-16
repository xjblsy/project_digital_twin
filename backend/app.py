from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 导入优化后的预测器
from predictor import FloodRiskPredictor

# 全局预测器实例
predictor = FloodRiskPredictor()
if not predictor.load_model():
    logger.error("❌ 模型加载失败")
    raise RuntimeError("模型加载失败")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_loaded,
        'timestamp': datetime.now().isoformat(),
        'service': 'Flood Risk Prediction API'
    })

@app.route('/predict', methods=['POST'])
def predict_risk():
    """
    风险预测接口
    
    请求体:
    {
        "historical_data": [
            {"date": "2025-01-01", "water_level": 52.5, "precipitation": 10, "temperature": 25, "humidity": 70},
            ...
        ]
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
        
        # 执行预测
        result = predictor.predict(df_history)
        result['success'] = True
        
        logger.info(f"✅ 预测成功 | 日期: {result['prediction_date']} | 风险: {result['risk_label']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ 预测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        data = request.get_json()
        
        if not data or 'datasets' not in data:
            return jsonify({
                'success': False,
                'error': '缺少 datasets 字段'
            }), 400
        
        results = []
        for i, dataset in enumerate(data['datasets']):
            try:
                df_history = pd.DataFrame(dataset)
                result = predictor.predict(df_history)
                result['dataset_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'dataset_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"❌ 批量预测失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
