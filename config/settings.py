"""
FallGuard AI - Configuration Settings
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class BaseConfig:
    SECRET_KEY = os.getenv('SECRET_KEY', 'fallguard-secret-key-change-in-production')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "fallguard.db")}')

    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True

    # SocketIO
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25

    # Detection Settings
    FALL_CONFIDENCE_THRESHOLD = 0.85
    ALERT_COOLDOWN_SECONDS = 30
    INACTIVITY_THRESHOLD_SECONDS = 5
    VELOCITY_THRESHOLD = 0.15
    OVERLAP_THRESHOLD = 0.5
    FPS_TARGET = 25

    # Weights for confidence scoring
    WEIGHT_POSTURE = 0.30
    WEIGHT_VELOCITY = 0.25
    WEIGHT_HEIGHT_CHANGE = 0.20
    WEIGHT_INACTIVITY = 0.15
    WEIGHT_OBJECT_CONTEXT = 0.10

    # Email (optional)
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', 'fallguard@example.com')

    # Twilio SMS (optional)
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
    TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '')

    # Storage
    MAX_EVENT_IMAGES = 1000
    EVENT_IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'events')
    EXPORT_DIR = os.path.join(BASE_DIR, 'static', 'exports')


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False


class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    FALL_CONFIDENCE_THRESHOLD = 0.88


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
