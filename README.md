# FallGuard AI — Healthcare Fall Detection Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A **production-ready** AI healthcare monitoring platform for real-time fall detection using a 4-stage multi-factor pipeline — engineered to maximise accuracy and minimise false positives.

---

## Key Features

| Feature | Details |
|---|---|
| **4-Stage AI Pipeline** | Human detection → Pose estimation → Multi-factor analysis → Context awareness |
| **Bed/Sofa Suppression** | YOLOv8 detects resting surfaces; bounding box overlap prevents false falls |
| **5-Factor Confidence** | Posture 30% · Velocity 25% · Height Δ 20% · Inactivity 15% · Context 10% |
| **Real-Time SocketIO** | Live annotated video feed + instant stats at 20–30 FPS |
| **Multi-Alert System** | Sound + Browser push + Email (SMTP) + SMS (Twilio) |
| **Healthcare Dashboard** | Glassmorphism UI · Dark/Light mode · Responsive |
| **Analytics** | Chart.js charts: daily falls, activity distribution, AI insights |
| **Export** | CSV event export · PDF report generation (WeasyPrint) |
| **Role-Based Auth** | Admin / Caregiver / Viewer with session management |
| **Docker Ready** | Dockerfile + docker-compose for one-command deployment |

---

## Architecture

```
fallguard/
├── app.py                      # Flask application factory
├── run.py                      # Entry point (starts detector + camera)
├── config/
│   └── settings.py             # Dev / Prod / Test configurations
├── app/
│   ├── models/
│   │   ├── database.py         # SQLAlchemy init + DB seeding
│   │   ├── user.py             # User model (RBAC)
│   │   ├── event.py            # Event + Alert models
│   │   ├── contact.py          # Emergency contacts
│   │   └── settings_model.py   # Per-user settings
│   ├── routes/
│   │   ├── auth.py             # Login / Signup / Logout
│   │   ├── dashboard.py        # Monitor + History pages
│   │   ├── api.py              # REST API (camera, events, export)
│   │   ├── analytics.py        # Analytics data endpoint
│   │   ├── alerts.py           # Alert history + resolve
│   │   └── settings.py         # Settings + contacts CRUD
│   └── services/
│       ├── fall_detector.py    # ← Core AI engine
│       ├── camera_service.py   # Webcam capture + SocketIO broadcast
│       ├── alert_service.py    # Email + SMS alert dispatch
│       └── report_service.py   # PDF + CSV generation
├── static/
│   ├── css/main.css            # Full design system
│   ├── js/main.js              # SocketIO + UI logic
│   └── events/                 # Saved fall snapshots
└── templates/
    ├── base.html               # Sidebar layout
    └── pages/
        ├── login.html
        ├── signup.html
        ├── monitor.html        # Live dashboard
        ├── history.html
        ├── alerts.html
        ├── analytics.html
        ├── settings.html
        └── about.html
```

---

## Detection Pipeline (Detailed)

### Stage 1 — Human Detection
- YOLOv8n runs at full frame speed; person bounding box tracked frame-to-frame.
- Fallback: OpenCV MOG2 background subtractor if ultralytics is not installed.

### Stage 2 — Pose Estimation (MediaPipe)
- 33 full-body landmarks extracted per frame.
- Key points used: shoulders, hips, ankles for spine vector calculation.

### Stage 3 — Multi-Factor Fall Analysis

| Factor | Logic | Weight |
|---|---|---|
| **Posture** | Body angle < 30° (horizontal) | 30% |
| **Velocity** | Sudden downward Δ of centre-of-mass Y | 25% |
| **Height Change** | Bounding box height drops ≥ 25% vs. baseline | 20% |
| **Inactivity** | Person still for ≥ 5s after suspected fall | 15% |
| **Object Context** | Floor vs. resting surface overlap logic | 10% |

**Alert fires only when total confidence ≥ 85%.**

### Stage 4 — Context Awareness (Anti-False-Positive)
```
IF person_bbox overlaps bed_bbox > 50%:
    surface = "bed"
    context_score = 0.0      ← kills fall confidence
    activity = "sleeping"

ELIF person_bbox overlaps couch_bbox > 50%:
    surface = "couch"
    context_score = 0.0
    activity = "lying_down"

ELSE:
    surface = "floor"
    context_score = 0.8      ← supports fall hypothesis
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Webcam

### 1. Clone & install
```bash
git clone https://github.com/you/fallguard-ai
cd fallguard-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set SECRET_KEY, email/SMS credentials if needed
```

### 3. Run
```bash
python run.py
```

Open **http://localhost:5000** → login with `admin / admin123`

---

## Docker Deployment
```bash
cp .env.example .env   # fill in your values
docker-compose up --build
```

---

## Production Deployment (Gunicorn + Nginx)

```bash
# Start with gunicorn + eventlet
gunicorn --worker-class eventlet -w 1 \
         --bind 0.0.0.0:5000 \
         "app:create_app('production')"
```

**Nginx config snippet:**
```nginx
location / {
    proxy_pass         http://127.0.0.1:5000;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";   # required for SocketIO
    proxy_set_header   Host $host;
}
```

---

## Database Schema

```sql
users              (id, username, email, password_hash, role, full_name, phone, ...)
events             (id, event_id UUID, event_type, activity_label, is_fall,
                    confidence_total, confidence_posture, confidence_velocity,
                    confidence_height, confidence_inactivity, confidence_context,
                    body_angle, velocity, detected_objects JSON, image_path, timestamp)
alerts             (id, event_id FK, alert_type, status, sent_at, recipient)
emergency_contacts (id, user_id FK, name, relationship, phone, email, notify_sms, notify_email)
user_settings      (id, user_id FK, confidence_threshold, alert_cooldown, enable_sound,
                    enable_sms, enable_email, dark_mode, camera_index, ...)
```

---

## Configuration Reference

All settings live in `config/settings.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `FALL_CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence to trigger alert |
| `ALERT_COOLDOWN_SECONDS` | `30` | Minimum seconds between alerts |
| `INACTIVITY_THRESHOLD_SECONDS` | `5` | Seconds of stillness for inactivity score |
| `VELOCITY_THRESHOLD` | `0.15` | Normalised downward velocity threshold |
| `OVERLAP_THRESHOLD` | `0.5` | Bounding box overlap % to classify as resting |
| `WEIGHT_POSTURE` | `0.30` | Posture factor weight |
| `WEIGHT_VELOCITY` | `0.25` | Velocity factor weight |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/camera/start` | Start camera capture |
| POST | `/api/camera/stop` | Stop camera |
| GET  | `/api/camera/status` | Camera + FPS status |
| POST | `/api/camera/switch` | Switch camera index |
| GET  | `/api/events` | Paginated event list |
| GET  | `/api/export/csv` | Download events as CSV |
| GET  | `/api/export/pdf` | Download PDF report |
| GET  | `/analytics/data` | JSON analytics data for charts |
| POST | `/alerts/resolve/<id>` | Mark alert as resolved |

---

## Roles & Permissions

| Permission | Admin | Caregiver | Viewer |
|---|---|---|---|
| View dashboard | ✅ | ✅ | ✅ |
| Edit settings | ✅ | ✅ | ❌ |
| Export data | ✅ | ✅ | ❌ |
| Manage users | ✅ | ❌ | ❌ |
| Delete events | ✅ | ❌ | ❌ |

---

## Performance Tips

- **CPU only**: YOLOv8n is the lightest model; target 20–25 FPS on modern laptops.
- **Raspberry Pi 4**: Disable YOLO (`YOLO_ENABLED=false`) and use pose-only mode (~15 FPS).
- **GPU**: Install `ultralytics[gpu]` and CUDA drivers for 30+ FPS with full object detection.
- **Frame skip**: For slow devices, process every 2nd frame in `camera_service.py`.

---

## License

MIT — free for personal, academic, and commercial use.
