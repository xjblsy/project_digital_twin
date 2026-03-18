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

# 模型初始化 - 改进版，支持多种部署环境
def get_model_directory():
    """获取模型目录，支持多种环境"""
    # 优先使用环境变量
    env_model_dir = os.environ.get('MODEL_DIR')
    if env_model_dir:
        logger.info(f"使用环境变量指定的模型目录: {env_model_dir}")
        return env_model_dir
    
    # 尝试常见的云端部署路径
    cloud_paths = [
        '/app/models',           # Docker容器默认路径
        '/workspace/models',     # 某些云平台路径
        '/home/app/models'       # 其他可能的路径
    ]
    
    for path in cloud_paths:
        if os.path.exists(path):
            logger.info(f"在云端路径找到模型目录: {path}")
            return path
    
    # 本地开发路径（相对路径）
    local_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    logger.info(f"使用本地模型目录: {local_path}")
    return local_path

model_dir = get_model_directory()

try:
    predictor = FloodRiskPredictor(model_dir=model_dir)
    if predictor.load_model():
        logger.info("✅ 模型加载成功")
        MODEL_LOADED = True
    else:
        logger.error(f"❌ 模型加载失败，检查路径: {model_dir}")
        MODEL_LOADED = False
except Exception as e:
    logger.error(f"模型初始化异常：{e}")
    logger.error(f"尝试的模型路径: {model_dir}")
    MODEL_LOADED = False
    predictor = None


@app.route('/', methods=['GET'])
def index():
    """根路由 - 返回 API 信息"""
    return jsonify({
        'service': 'Flood Risk Prediction API',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'batch_predict': 'POST /batch_predict'
        },
        'model_loaded': MODEL_LOADED,
        'model_path': model_dir,  # 显示当前使用的模型路径
        'status': 'running'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'model_path': model_dir,  # 显示当前使用的模型路径
        'timestamp': datetime.now().isoformat(),
        'service': 'Flood Risk Prediction API'
    })


@app.route('/predict', methods=['POST'])
def predict_risk():
    """
    风险预测接口
    """
    # 检查模型是否已加载
    if not MODEL_LOADED or predictor is None:
        return jsonify({
            'success': False,
            'error': f'模型未加载，请检查服务器状态。尝试路径: {model_dir}'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'success': False,
                'error': '缺少 historical_data 字段'
            }), 400
        
        df_history = pd.DataFrame(data['historical_data'])
        
        # 数据验证
        required_cols = ['date', 'water_level']
        missing_cols = [col for col in required_cols if col not in df_history.columns]
        if missing_cols:
            return jsonify({
                'success': False,
                'error': f'缺少必要列：{missing_cols}'
            }), 400

        if len(df_history) < 14:
            return jsonify({
                'success': False,
                'error': f'历史数据不足，至少需要 14 天，当前只有 {len(df_history)} 天'
            }), 400
        
        # 执行预测
        result = predictor.predict(df_history)
        result['success'] = True
        
        logger.info(f"✅ 预测成功 | 风险等级: {result['risk_level']} | 风险值: {result['risk']}")
        
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
    # ✅ 添加模型检查
    if not MODEL_LOADED or predictor is None:
        return jsonify({
            'success': False,
            'error': f'模型未加载，请检查服务器状态。尝试路径: {model_dir}'
        }), 500
    
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
                
                # 对每个数据集也进行基本验证
                required_cols = ['date', 'water_level']
                missing_cols = [col for col in required_cols if col not in df_history.columns]
                if missing_cols:
                    results.append({
                        'dataset_index': i,
                        'success': False,
                        'error': f'缺少必要列：{missing_cols}'
                    })
                    continue

                if len(df_history) < 14:
                    results.append({
                        'dataset_index': i,
                        'success': False,
                        'error': f'历史数据不足，至少需要 14 天，当前只有 {len(df_history)} 天'
                    })
                    continue
                
                result = predictor.predict(df_history)
                result['dataset_index'] = i
                result['success'] = True
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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
