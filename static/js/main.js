/* ═══════════════════════════════════════════════════════════════
   FallGuard AI — Main JavaScript
   Handles: SocketIO, camera feed, live stats, alerts, theme
═══════════════════════════════════════════════════════════════ */

// ── Global State ──────────────────────────────────────────────
const FG = {
  socket: null,
  isMonitoring: false,
  currentActivity: 'unknown',
  currentConf: 0,
  unreadAlerts: 0,
  currentEventId: null,
  alertSound: null,
  lastAlertTime: 0,
  theme: localStorage.getItem('fg_theme') || 'dark'
};

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(FG.theme);
  initSocket();
  initUI();
  loadAlertBadge();
  checkDetectionReady();
});

async function checkDetectionReady() {
  try {
    const r = await fetch('/api/detection/status');
    const d = await r.json();
    if (!d.ready) {
      showToast('⚠ ' + (d.error || 'Detection model not loaded. Place pose_landmarker.task in the project root.'), 'error', 10000);
      setSystemStatus('error', 'Model missing');
    } else {
      setSystemStatus('online', 'Detection ready');
    }
  } catch(e) { /* endpoint optional */ }
}

// ── SocketIO ──────────────────────────────────────────────────
function initSocket() {
  FG.socket = io({ transports: ['websocket', 'polling'] });

  FG.socket.on('connect', () => {
    console.log('[FallGuard] Socket connected');
    setSystemStatus('online', 'Connected');
  });

  FG.socket.on('disconnect', () => {
    setSystemStatus('offline', 'Offline');
    setNavLive(false);
  });

  FG.socket.on('frame_update', (data) => {
    updateCameraFeed(data);
    updateLiveStats(data);
  });

  FG.socket.on('detection_error', (data) => {
    console.error('Detection error:', data.message);
    setSystemStatus('error', 'Detection error — check server logs');
    showToast('⚠ ' + data.message, 'error');
    const ph = document.getElementById('cameraPlaceholder');
    if (ph) {
      let errEl = document.getElementById('cameraErrorMsg');
      if (!errEl) {
        errEl = document.createElement('p');
        errEl.id = 'cameraErrorMsg';
        errEl.style.cssText = 'color:#f85149;font-size:12px;max-width:320px;text-align:center;margin-top:8px';
        ph.appendChild(errEl);
      }
      errEl.textContent = data.message;
    }
  });

  // Video file finished playing → reset UI automatically
  // Clip ready notification
  FG.socket.on('fall_clip_ready', (data) => {
    showToast(
      `📹 Fall clip saved — <a href="/${data.clip_path}" target="_blank" style="color:#fff;text-decoration:underline">View clip</a>`,
      'info', 6000
    );
  });

  FG.socket.on('video_ended', (data) => {
    if (typeof _resetCameraUI === 'function') _resetCameraUI();
    setSystemStatus('offline', 'Video ended');
    showToast(data.message || 'Video playback complete', 'info');
  });

  FG.socket.on('fall_alert', (data) => {
    handleFallAlert(data);
  });
}

// ── Camera Feed ───────────────────────────────────────────────
function updateCameraFeed(data) {
  const img = document.getElementById('cameraFeed');
  const placeholder = document.getElementById('cameraPlaceholder');

  if (img && data.frame) {
    img.src = `data:image/jpeg;base64,${data.frame}`;
    img.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
    FG.isMonitoring = true;
    setNavLive(true);
  }

  // Update live badge
  const liveBadge = document.getElementById('liveBadge');
  if (liveBadge) {
    if (data.is_fall) {
      liveBadge.textContent = '⚠ FALL';
      liveBadge.className = 'camera-badge fall';
    } else {
      liveBadge.textContent = '● LIVE';
      liveBadge.className = 'camera-badge live';
    }
  }

  // FPS
  const fpsBadge = document.getElementById('fpsBadge');
  if (fpsBadge) fpsBadge.textContent = `${data.fps || 0} FPS`;
}

// ── Live Stats ────────────────────────────────────────────────
function updateLiveStats(data) {
  // Activity
  const actLabel = document.getElementById('activityLabel');
  const actIcon  = document.getElementById('activityIcon');
  const actWrap  = document.getElementById('activityIconWrap');
  const actSub   = document.getElementById('activitySublabel');

  if (actLabel) actLabel.textContent = formatActivity(data.activity);
  if (actSub)   actSub.textContent   = data.is_fall ? '⚠ Alert triggered' : `Surface: ${data.context_surface || 'floor'}`;

  if (actIcon && actWrap) {
    actIcon.className = `fa-solid ${activityIcon(data.activity)}`;
    actWrap.className = `activity-icon-wrap ${data.is_fall ? 'fall-state' : ''}`;
  }

  // Total confidence
  setConfBar('confTotal', data.confidence, data.is_fall);

  // Individual bars
  setConfBar('confPosture',    data.conf_posture);
  setConfBar('confVelocity',   data.conf_velocity);
  setConfBar('confHeight',     data.conf_height);
  setConfBar('confInactivity', data.conf_inactivity);
  setConfBar('confContext',    data.conf_context);

  // Angle / velocity values
  setText('valAngle',    data.body_angle ? `${data.body_angle.toFixed(1)}°` : '—');
  setText('valVelocity', data.velocity   ? data.velocity.toFixed(3) : '—');
  setText('valSurface',  data.context_surface || '—');
  setText('valObjects',  (data.detected_objects || []).join(', ') || 'none');
}

function setConfBar(id, value, isDanger = false) {
  const pct = Math.round((value || 0) * 100);
  const bar = document.getElementById(id + 'Bar');
  const val = document.getElementById(id + 'Val');
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.className   = `conf-bar ${isDanger || pct >= 85 ? 'danger' : ''}`;
  }
  if (val) val.textContent = `${pct}%`;
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ── Fall Alert ────────────────────────────────────────────────
function handleFallAlert(data) {
  const now = Date.now();
  if (now - FG.lastAlertTime < 5000) return;  // client-side debounce
  FG.lastAlertTime = now;
  FG.currentEventId = data.event_id;

  // Increment badge
  FG.unreadAlerts++;
  updateBellBadge();

  // Show modal
  const details = document.getElementById('alertDetails');
  if (details) {
    const ts = new Date(data.timestamp).toLocaleString();
    details.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        <div class="alert-detail-item"><div class="detail-label">Event ID</div><div class="detail-val font-mono">${data.event_id}</div></div>
        <div class="alert-detail-item"><div class="detail-label">Time</div><div class="detail-val">${ts}</div></div>
        <div class="alert-detail-item"><div class="detail-label">Activity</div><div class="detail-val">${formatActivity(data.activity)}</div></div>
        <div class="alert-detail-item"><div class="detail-label">Confidence</div><div class="detail-val text-danger">${Math.round((data.confidence||0)*100)}%</div></div>
      </div>`;
  }

  const confFill  = document.getElementById('alertConfFill');
  const confValue = document.getElementById('alertConfValue');
  if (confFill)  confFill.style.width  = `${Math.round((data.confidence||0)*100)}%`;
  if (confValue) confValue.textContent = `${Math.round((data.confidence||0)*100)}%`;

  showModal('fallAlertModal');

  // Sound
  playAlertSound();

  // Browser notification
  sendBrowserNotification(data);

  // Update status dot to alert
  setSystemStatus('alert', 'FALL DETECTED');

  // Update recent events list
  addRecentEvent(data);
}

function resolveAlert() {
  if (!FG.currentEventId) { dismissModal('fallAlertModal'); return; }
  fetch(`/alerts/resolve/${FG.currentEventId}`, { method: 'POST' })
    .catch(e => console.warn('Resolve failed', e));
  dismissModal('fallAlertModal');
  setSystemStatus('online', 'Monitoring');
}

// ── Alert Sound ───────────────────────────────────────────────
function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [0, 0.3, 0.6].forEach(delay => {
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.3, ctx.currentTime + delay);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.25);
      osc.start(ctx.currentTime + delay);
      osc.stop(ctx.currentTime + delay + 0.25);
    });
  } catch (e) { console.warn('Audio not available'); }
}

// ── Browser Notifications ──────────────────────────────────────
function sendBrowserNotification(data) {
  if (!('Notification' in window)) return;
  if (Notification.permission === 'granted') {
    new Notification('🚨 FallGuard — Fall Detected!', {
      body: `Confidence: ${Math.round((data.confidence||0)*100)}% | ${new Date().toLocaleTimeString()}`,
      icon: '/static/images/icon.png',
      tag: 'fall-alert'
    });
  } else if (Notification.permission !== 'denied') {
    Notification.requestPermission().then(perm => {
      if (perm === 'granted') sendBrowserNotification(data);
    });
  }
}

// ── Recent Events ──────────────────────────────────────────────
function addRecentEvent(data) {
  const list = document.getElementById('recentEventsList');
  if (!list) return;
  const ts = new Date(data.timestamp).toLocaleTimeString();
  const item = document.createElement('div');
  item.className = 'recent-event-item';
  item.innerHTML = `
    <div class="event-dot danger"></div>
    <div class="event-info">
      <div class="event-type">Fall Detected</div>
      <div class="event-time">${ts} — Conf: ${Math.round((data.confidence||0)*100)}%</div>
    </div>`;
  list.insertBefore(item, list.firstChild);
  // Keep max 5
  while (list.children.length > 5) list.removeChild(list.lastChild);
}

// ── Camera Control ────────────────────────────────────────────
async function startCamera(idx = 0) {
  const btn = document.getElementById('startCameraBtn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner spin"></i> Starting...'; }
  try {
    const res = await fetch('/api/camera/start', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ camera_index: idx })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.message);
    showToast('Camera started', 'success');
    if (btn) { btn.textContent = ''; btn.innerHTML = '<i class="fa-solid fa-video-slash"></i> Close Camera'; btn.onclick = stopCamera; }
  } catch (e) {
    showToast(`Camera error: ${e.message}`, 'error');
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-play"></i> Start'; }
  }
}

async function stopCamera() {
  await fetch('/api/camera/stop', { method: 'POST' });
  const img = document.getElementById('cameraFeed');
  const placeholder = document.getElementById('cameraPlaceholder');
  if (img) { img.src = ''; img.style.display = 'none'; }
  if (placeholder) placeholder.style.display = 'flex';
  FG.isMonitoring = false;
  setNavLive(false);
  setSystemStatus('offline', 'Camera stopped');
  showToast('Camera stopped', 'info');
  const btn = document.getElementById('startCameraBtn');
  if (btn) { btn.innerHTML = '<i class="fa-solid fa-play"></i> Start'; btn.onclick = () => startCamera(); btn.disabled = false; }
}

// ── UI Helpers ────────────────────────────────────────────────
function showModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove('hidden');
}
function dismissModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add('hidden');
}

function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer') || document.querySelector('.page-content');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  const icon = type === 'error' ? 'fa-circle-xmark' : type === 'success' ? 'fa-circle-check' : 'fa-circle-info';
  t.innerHTML = `<i class="fa-solid ${icon}"></i>${message}`;
  container.insertBefore(t, container.firstChild);
  setTimeout(() => t.remove(), duration);
}

function setSystemStatus(state, text) {
  const dot  = document.querySelector('.status-dot');
  const span = document.querySelector('.system-status span');
  if (dot)  { dot.className  = `status-dot ${state}`; }
  if (span) { span.textContent = text; }
}

function setNavLive(active) {
  const dot = document.getElementById('navLiveDot');
  if (dot) dot.classList.toggle('active', active);
}

function updateBellBadge() {
  const badge = document.getElementById('bellBadgeCount');
  const navBadge = document.getElementById('navAlertBadge');
  if (FG.unreadAlerts > 0) {
    if (badge)    { badge.textContent = FG.unreadAlerts; badge.classList.remove('hidden'); }
    if (navBadge) { navBadge.textContent = FG.unreadAlerts; }
  }
}

async function loadAlertBadge() {
  try {
    const r = await fetch('/api/events?falls_only=true&limit=1');
    // Just check total
    const d = await r.json();
    // Count unresolved from server if needed
  } catch(e) {}
}

// ── Theme ──────────────────────────────────────────────────────
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = document.getElementById('themeIcon');
  if (icon) icon.className = `fa-solid ${theme === 'dark' ? 'fa-moon' : 'fa-sun'}`;
  localStorage.setItem('fg_theme', theme);
  FG.theme = theme;
}

function toggleTheme() {
  applyTheme(FG.theme === 'dark' ? 'light' : 'dark');
}

// ── Activity Helpers ──────────────────────────────────────────
function formatActivity(activity) {
  const map = {
    standing: 'Standing', walking: 'Walking', sitting: 'Sitting',
    bending: 'Bending', lying: 'Lying Down', sleeping: 'Sleeping',
    lying_down: 'Lying on Sofa', falling: 'Falling!', fallen: 'Fallen!',
    unknown: 'Detecting…', no_person: 'No Person'
  };
  return map[activity] || activity;
}

function activityIcon(activity) {
  const map = {
    standing: 'fa-person', walking: 'fa-person-walking', sitting: 'fa-chair',
    bending: 'fa-person-digging', lying: 'fa-bed', sleeping: 'fa-bed',
    lying_down: 'fa-couch', falling: 'fa-person-falling', fallen: 'fa-person-falling-burst',
    unknown: 'fa-circle-question', no_person: 'fa-eye-slash'
  };
  return map[activity] || 'fa-person';
}

// ── Init UI ───────────────────────────────────────────────────
function initUI() {
  // Theme toggle
  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

  // Mobile menu
  const menuBtn = document.getElementById('mobileMenuBtn');
  const sidebar = document.getElementById('sidebar');
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
  }

  // Notification permission
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }

  // Alert bell click → go to alerts
  const bell = document.getElementById('alertBell');
  if (bell) bell.addEventListener('click', () => {
    window.location.href = '/alerts/';
    FG.unreadAlerts = 0;
    updateBellBadge();
  });
}
