"""
FallGuard AI - Database init helper
db instance lives in app/__init__.py to avoid circular imports.
"""
from app import db


def init_db():
    """Create all tables and seed default admin."""
    # Import models so SQLAlchemy knows about them
    from app.models.user          import User
    from app.models.event         import Event, Alert
    from app.models.contact       import EmergencyContact
    from app.models.settings_model import UserSettings
    from app.models.camera_node   import CameraNode

    db.create_all()
    _seed_default_admin()


def _seed_default_admin():
    from app.models.user import User
    from werkzeug.security import generate_password_hash
    if User.query.count() == 0:
        admin = User(
            username='admin',
            email='admin@fallguard.ai',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            full_name='System Administrator',
            is_active=True
        )
        db.session.add(admin)
        db.session.commit()
        print("  [DB] Default admin created: admin / admin123")
