"""
app/services/detection/__init__.py
Makes `from app.services.detection import detection_config as cfg` work,
and exposes the public API of the detection sub-package.
"""
from app.services.detection import detection_config  # noqa: F401
