"""
camera_node.py – Database model for registered camera devices.
Each row = one ward/room device on the network.
"""

import uuid
from datetime import datetime
from app.models.database import db


class CameraNode(db.Model):
    __tablename__ = 'camera_nodes'

    id         = db.Column(db.String(36), primary_key=True,
                           default=lambda: str(uuid.uuid4()))
    name       = db.Column(db.String(100), nullable=False)   # e.g. "Ward 1"
    location   = db.Column(db.String(200), default="")       # e.g. "Room 101"
    host       = db.Column(db.String(200), nullable=True)    # e.g. "192.168.1.10" (remote nodes only)
    port       = db.Column(db.Integer, default=5000)
    enabled    = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # -- local stream support (e.g. a phone running an IP-camera app) --
    # If is_local is True, this node is NOT a separate FallGuard server.
    # Instead THIS server opens `stream_url` directly (like any webcam) and
    # runs its own detector against it. host/port are unused in that case.
    is_local   = db.Column(db.Boolean, default=False)
    stream_url = db.Column(db.String(500), default="")       # e.g. "http://192.168.1.42:8080/video"
    is_running = db.Column(db.Boolean, default=False)        # local capture currently active?

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def socket_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {
            'id':         self.id,
            'name':       self.name,
            'location':   self.location,
            'host':       self.host,
            'port':       self.port,
            'url':        self.url,
            'enabled':    self.enabled,
            'is_local':   self.is_local,
            'stream_url': self.stream_url,
            'is_running': self.is_running,
        }