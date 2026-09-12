"""
FallGuard AI - Event & Alert Models
"""

from datetime import datetime
from app import db
import uuid


class Event(db.Model):
    __tablename__ = 'events'

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # Classification
    event_type = db.Column(db.String(50), nullable=False)  # fall, sleeping, sitting, standing, etc.
    activity_label = db.Column(db.String(50), nullable=True)
    is_fall = db.Column(db.Boolean, default=False)

    # Confidence scores
    confidence_total = db.Column(db.Float, default=0.0)
    confidence_posture = db.Column(db.Float, default=0.0)
    confidence_velocity = db.Column(db.Float, default=0.0)
    confidence_height = db.Column(db.Float, default=0.0)
    confidence_inactivity = db.Column(db.Float, default=0.0)
    confidence_context = db.Column(db.Float, default=0.0)

    # Context
    detected_objects = db.Column(db.Text, nullable=True)  # JSON string
    body_angle = db.Column(db.Float, nullable=True)
    velocity = db.Column(db.Float, nullable=True)

    # Media
    image_path = db.Column(db.String(255), nullable=True)

    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    is_resolved = db.Column(db.Boolean, default=False)

    # Alert
    alert = db.relationship('Alert', backref='event', uselist=False, lazy=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'event_id': self.event_id,
            'event_type': self.event_type,
            'activity_label': self.activity_label,
            'is_fall': self.is_fall,
            'confidence_total': round(self.confidence_total, 3),
            'confidence_posture': round(self.confidence_posture, 3),
            'confidence_velocity': round(self.confidence_velocity, 3),
            'confidence_height': round(self.confidence_height, 3),
            'confidence_inactivity': round(self.confidence_inactivity, 3),
            'confidence_context': round(self.confidence_context, 3),
            'body_angle': self.body_angle,
            'velocity': self.velocity,
            'detected_objects': self.detected_objects,
            'image_path': self.image_path,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'is_resolved': self.is_resolved
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('events.id'), nullable=False)
    alert_type = db.Column(db.String(30), nullable=False)  # browser, sms, email, sound
    status = db.Column(db.String(20), default='pending')  # pending, sent, failed
    sent_at = db.Column(db.DateTime, nullable=True)
    recipient = db.Column(db.String(150), nullable=True)
    message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'event_id': self.event_id,
            'alert_type': self.alert_type,
            'status': self.status,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'recipient': self.recipient
        }
