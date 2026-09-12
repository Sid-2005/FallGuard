"""
FallGuard AI — Flask application factory (lives inside the app package).
"""

import os
import logging
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Global extension instances (created here, bound to app in create_app)
db         = SQLAlchemy()
socketio   = SocketIO()
login_manager = LoginManager()

logger = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    """Application factory."""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    # Build path relative to THIS file (app/__init__.py)
    app_dir     = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(app_dir)

    app = Flask(
        __name__,
        template_folder=os.path.join(project_dir, 'templates'),
        static_folder=os.path.join(project_dir, 'static')
    )

    # Load config
    from config.settings import config as cfg_map
    app.config.from_object(cfg_map[config_name])

    # Ensure directories
    os.makedirs(os.path.join(project_dir, 'logs'),           exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'static', 'events'),  exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'static', 'exports'), exist_ok=True)

    # Set absolute paths in config
    app.config['EVENT_IMAGE_DIR'] = os.path.join(project_dir, 'static', 'events')
    app.config['EXPORT_DIR']      = os.path.join(project_dir, 'static', 'exports')

    # Add file log handler now that we know project_dir
    fh = logging.FileHandler(os.path.join(project_dir, 'logs', 'fallguard.log'), encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(fh)

    # Init extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*", async_mode='threading')
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.query.get(int(user_id))

    # Register blueprints
    from app.routes.auth      import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.api       import api_bp
    from app.routes.analytics import analytics_bp
    from app.routes.settings  import settings_bp
    from app.routes.alerts    import alerts_bp
    from app.routes.central   import central_bp

    app.register_blueprint(auth_bp,       url_prefix='/auth')
    app.register_blueprint(dashboard_bp,  url_prefix='/')
    app.register_blueprint(api_bp,        url_prefix='/api')
    app.register_blueprint(analytics_bp,  url_prefix='/analytics')
    app.register_blueprint(settings_bp,   url_prefix='/settings')
    app.register_blueprint(alerts_bp,     url_prefix='/alerts')
    app.register_blueprint(central_bp)

    with app.app_context():
        from app.models.database import init_db
        init_db()

    logger.info(f"FallGuard AI started [{config_name}]")
    return app
