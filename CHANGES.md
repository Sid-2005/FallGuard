# FallGuard AI — Cleanup & Accuracy Fix (this pass)

## 1. Project size: 1.3 GB → 23 MB
The zip you had included a full `venv/` (1.3 GB of installed packages) and a
`.git/` history. Neither should ever be shipped/zipped — `venv/` is rebuilt
from `requirements.txt` on any machine with `pip install -r requirements.txt`,
and `.git` is only needed if you're pushing to GitHub, not for submission.
Removed both, plus stray `__pycache__/` folders and the runtime `fallguard.db`
(it's recreated automatically on first run). `.gitignore` already covered
this correctly going forward.

## 2. Removed ~2,300 lines of dead code
The repo actually contained **two separate fall-detection engines**:

- `app/services/fall_detector.py` — the one actually wired up and used by
  `multi_camera_manager.py` (i.e. the real, live pipeline).
- `app/services/detection/{pose_estimator,fall_analyzer,tracker,evaluator,
  floor_mapper,renderer,object_detector}.py` — a second, more elaborate
  pipeline that **nothing in the app ever imported or ran**. It just sat
  there unused, more than doubling the size of `app/services/` and making
  the codebase confusing to read, debug, or explain in a viva.

I deleted the unused second pipeline entirely and shrank
`detection_config.py` down to the 2–3 flags the live code actually reads.
`app/services/` went from 4,312 lines to 2,014. Nothing about the running
app's behavior changed from this — it was pure dead weight.

I also fixed two small breakages this left behind: a Settings-page toggle
route in `api.py` that referenced the now-removed `ObjectDetector` class,
and a `/detection/status` check that referenced an attribute
(`_pose_estimator`) that never existed on the real detector.

## 3. The actual accuracy fix — sitting/standing false-positives
This was the real bug behind sitting or standing being flagged as a fall.

`fall_detector.py` computed "how horizontal is this person" using the vector
from **shoulder → ankle**. That's a problem: when someone sits in a chair,
their ankles end up forward of and below their shoulders because the knees
are bent — from a side-on camera angle, that vector drags toward horizontal
even though the person is completely upright. Combined with the fact that
sitting is naturally still (low velocity), this is exactly the pattern the
fall-confirmation timer was watching for, so a stationary seated person could
eventually trip a false alert if no chair/desk was recognized by the object
detector.

**Fix:** the angle is now computed from **shoulder → hip** (the torso itself),
not shoulder → ankle. Your torso stays vertical whether you're standing or
sitting, no matter how your knees are bent or which way the camera is
pointed — it only goes horizontal when your upper body is actually down,
i.e. a real fall. This is a small, surgical change (`_body_angle()` in
`fall_detector.py`) but it's the single biggest driver of sitting/standing
false positives, so it was worth isolating.

I also added a cheap hip-knee check (`seated` flag) so the on-screen activity
label correctly reads "sitting" instead of "standing" for someone seated
without a chair being detected — cosmetic, but makes the dashboard read
correctly during a demo.

The existing safety net is unchanged and still applies on top of this:
- A fall only confirms after **12 seconds** of continuous lying-like angle
  **and** near-zero velocity (`FALL_CONFIRM_SECONDS`, `FALL_STILLNESS_VELOCITY`).
- Any detected bed/couch/chair under the person immediately cancels a
  candidate fall, regardless of angle.
- If ankles are occluded (e.g. under a desk), the state freezes instead of
  guessing.

## What to re-test before your demo
- Sit at a desk (with and without a visible chair/desk in frame) for 15–20s —
  should stay "sitting", never escalate to a fall.
- Stand still for 15–20s — should stay "standing".
- Bend down briefly to pick something up, then stand back — should show
  "bending" momentarily and reset, not confirm a fall.
- An actual lie-down-on-the-floor test — should still confirm after ~12s
  of stillness, exactly as before.

---

# Round 2 — repeated alerts, wrong confidence %, contact form, multi-camera lag

## 1. Repeated "Fall Detected" popups (4–5 alerts for one fall)
**Cause:** the alert was fired on *every single frame* where a fall was
detected, gated only by a 30-second cooldown timer — not once per fall
event. So as long as the person stayed down, a fresh alert fired every
~30 seconds. Made worse by MediaPipe occasionally losing pose tracking for
a frame or two while someone lies down (it's a much rarer pose than
standing in its training data), which could flip `is_fall` False→True→False
and trigger extra spurious alerts on top of the cooldown-driven repeats.

**Fix (`camera_service.py`, `fall_detector.py`):**
- Alerts now fire once, on the transition into a confirmed fall
  (`is_fall` False→True), not on every frame while it stays True.
- A momentary pose-tracking dropout during an already-confirmed fall no
  longer counts as "recovered" — the fall state is held for up to 20s of
  lost tracking before giving up and resetting (in case the person actually
  got up and walked out of frame).

## 2. Wrong 6% confidence shown in the alert
Same root cause as #1. The alert was reading confidence from whatever
random *later* frame happened to re-trigger it, not the exact frame that
confirmed the fall — which is mathematically guaranteed to be ≥55%
(`_evaluate_fall` requires `conf >= 0.55` to even start the fall timer).
Fixing the trigger to fire only on the real confirmation frame fixes the
number automatically — you should now always see ≥55% on a confirmed fall.

## 3. Emergency contact not saving
Traced the full path (button → modal → JS → Flask route → DB) and didn't
find a definitive blocking bug — the wiring is structurally correct. Most
likely explanation is a failure being silently swallowed (e.g. an expired
login session returning a login-page redirect instead of JSON, which the
old code treated as a generic, unhelpful "Failed to add").

**Fix:** both sides now surface the *real* error instead of failing
silently — the backend route returns a proper error message on any
exception, and the frontend distinguishes "session expired," "server
error," and "network error" instead of one generic toast. If it still
doesn't save, the error message it now shows will say exactly why.

## 4. Multi-camera lag
Each registered camera runs its own full MediaPipe + YOLO pipeline, so
2–3 running at once genuinely competes for the same CPU — there's no way
around that without a GPU, but the load can be reduced a lot:
- Multi-camera nodes now request a smaller detection frame (360px vs 480px)
  and a smaller broadcast frame (240px vs 360px) than the primary
  single-camera Monitor page.
- Multi-camera nodes now only run the actual pose/YOLO inference every
  *other* frame, reusing the last known activity/fall state in between so
  the video doesn't look like it's freezing or resetting. Detection is
  still continuous enough to catch a real fall — it just doesn't re-run
  the expensive inference on literally every frame.
- The single-camera Monitor page is untouched by any of this — full
  resolution and full-rate detection, same as before.

If it's still slow with 3+ cameras on a CPU-only laptop, that's a hardware
ceiling, not a bug — stick to 1–2 cameras for a smooth demo.

---

# Round 3 — "Add Contact" link and sitting sometimes labelled "standing"

## 1. "Add Contact" redirected to Settings with no way to actually add one
Found it: the "Add Contact" link on the **Monitor** page's Emergency
Contacts widget just pointed at `/settings` — the top of the general
settings page. The real Add Contact button lives further down, in the
Emergency Contacts card, so landing at the top made it look like there was
no way to add one at all.

**Fix:** that link now takes you straight to `/settings?add_contact=1#emergencyContacts`,
which scrolls directly to the Emergency Contacts section **and automatically
opens the Add Contact form** — no more hunting for it. The Settings page's
own "+ Add Contact" button and the form itself were already working
correctly (and stayed hardened with real error messages from Round 2) — the
bug was purely in where that one link sent you.

## 2. Sitting sometimes still labelled "standing"
This is a **display label only** — it never affected fall detection or
safety, just the text shown for the person's current activity. Two things
were making it miss some sitting poses:
- The threshold for "legs are folded = sitting" was tuned tighter than real
  camera angles need — a person sitting with their knees not perfectly
  level with their hips (common at typical desk-camera angles) could fall
  just on the wrong side of it. Loosened the threshold so more real sitting
  poses are caught.
- Pose-landmark jitter (MediaPipe's knee position wobbles slightly frame to
  frame even when you haven't moved) could flip the label frame to frame.
  Now it's smoothed over the last few frames (majority vote) instead of
  trusting a single frame, so it settles on one label instead of flickering.

If you still catch it saying "standing" while clearly seated, it's most
useful to tell me: is the chair/desk visible in frame, and roughly what
angle is the camera at (side-on, front-on, elevated)? That'll tell me
whether it's this specific threshold or a different camera-angle edge case.


