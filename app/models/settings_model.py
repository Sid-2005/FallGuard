from app import db
from datetime import datetime

class UserSettings(db.Model):
    __tablename__ = 'user_settings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    confidence_threshold = db.Column(db.Float, default=0.85)
    alert_cooldown = db.Column(db.Integer, default=30)
    enable_sound = db.Column(db.Boolean, default=True)
    enable_browser_notif = db.Column(db.Boolean, default=True)
    enable_sms = db.Column(db.Boolean, default=False)
    enable_email = db.Column(db.Boolean, default=False)
    dark_mode = db.Column(db.Boolean, default=True)
    camera_index = db.Column(db.Integer, default=0)
    inactivity_threshold = db.Column(db.Integer, default=5)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'confidence_threshold': self.confidence_threshold,
            'alert_cooldown': self.alert_cooldown,
            'enable_sound': self.enable_sound,
            'enable_browser_notif': self.enable_browser_notif,
            'enable_sms': self.enable_sms,
            'enable_email': self.enable_email,
            'dark_mode': self.dark_mode,
            'camera_index': self.camera_index,
            'inactivity_threshold': self.inactivity_threshold
        }
