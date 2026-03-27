"""
MiroFish Backend - Flask应用工厂
"""

import os
import warnings

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS
from flask_caching import Cache

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 配置缓存
    cache_config = {
        'CACHE_TYPE': 'SimpleCache',  # 生产环境可改为 Redis
        'CACHE_DEFAULT_TIMEOUT': 300,  # 默认 5 分钟缓存
    }
    cache = Cache(app, config=cache_config)
    app.cache = cache  # 挂载到 app 上方便其他地方使用

    # 设置JSON编码：确保中文直接显示（而不是 \uXXXX 格式）
    # Flask >= 2.3 使用 app.json.ensure_ascii，旧版本使用 JSON_AS_ASCII 配置
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # 设置日志
    logger = setup_logger('mirofish')
    
    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend 启动中...")
        logger.info("=" * 50)
    
    # 启用CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 注册模拟进程清理函数（确保服务器关闭时终止所有模拟进程）
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已注册模拟进程清理函数")
    
    # 请求日志中间件
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"请求: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"请求体: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"响应: {response.status_code}")
        return response
    
    # 注册蓝图
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # 健康检查
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    # On startup: recover any projects stuck in graph_building (task was killed by restart)
    # TOFIX: 这里的恢复逻辑比较简单，依赖于 Neo4j 中是否有数据。理想情况下应该有更可靠的方式来判断任务是否真的完成（比如检查构建日志或使用更细粒度的状态）。但这个简单的检查至少可以避免大多数重启后卡在 graph_building 的情况。
    # if should_log_startup:
    #     _recover_stuck_projects()

    if should_log_startup:
        logger.info("MiroFish Backend 启动完成")

    return app


def _recover_stuck_projects():
    """Mark graph_building projects as completed if Neo4j already has their data."""
    from .models.project import ProjectManager, ProjectStatus
    from .utils.logger import get_logger as _get_logger
    _log = _get_logger('mirofish.startup')
    try:
        for p in ProjectManager.list_projects():
            if p.status == ProjectStatus.GRAPH_BUILDING and p.graph_id:
                from .services.graphiti_adapter import _get_graphiti, _run, _neo4j_query
                g = _get_graphiti()
                r = _run(_neo4j_query(g,
                    'MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS n',
                    {'gid': p.graph_id}
                ))
                node_count = int(r[0]['n']) if r else 0
                if node_count > 0:
                    p.status = ProjectStatus.GRAPH_COMPLETED
                    p.graph_build_task_id = None
                    ProjectManager.save_project(p)
                    _log.info(f"Recovered stuck project {p.project_id}: {node_count} nodes found, marked graph_completed")
    except Exception as e:
        _get_logger('mirofish.startup').warning(f"Startup recovery failed: {e}")

