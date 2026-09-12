"""
FallGuard AI - User Model
"""

from flask_login import UserMixin
from datetime import datetime
from app import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    role = db.Column(db.String(20), nullable=False, default='viewer')
    is_active = db.Column(db.Boolean, default=True)
    phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    events = db.relationship('Event', backref='created_by_user', lazy=True, foreign_keys='Event.user_id')
    settings = db.relationship('UserSettings', backref='user', uselist=False, lazy=True)
    contacts = db.relationship('EmergencyContact', backref='user', lazy=True)

    ROLES = {
        'admin': ['view', 'edit', 'delete', 'manage_users', 'export', 'settings'],
        'caregiver': ['view', 'edit', 'export', 'settings'],
        'viewer': ['view']
    }

    def has_permission(self, perm: str) -> bool:
        return perm in self.ROLES.get(self.role, [])

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'
