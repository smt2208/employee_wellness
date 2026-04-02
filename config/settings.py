# config/settings.py
# Central configuration for the wellness pipeline.
# All thresholds, model paths, and tunable parameters live here.
# Edit this file to change pipeline behaviour without touching core logic.

# ─── VIDEO INPUT ─────────────────────────────────────────────────────────────
VIDEO_SOURCE = 0
TARGET_FPS   = 5
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ─── YOLO PERSON DETECTION ───────────────────────────────────────────────────
YOLO_MODEL_PATH      = "yolov8n.pt"
YOLO_CONF_THRESHOLD  = 0.5
YOLO_IOU_THRESHOLD   = 0.4
YOLO_PERSON_CLASS_ID = 0

# ─── BOTSORT ─────────────────────────────────────────────────────────────────
# BoTSORT uses appearance-based Re-ID so it can re-identify a person
# when they return to the frame after a long absence (meetings, breaks etc.)
# This is critical for office environments where people frequently leave frame.
REID_MODEL_PATH = "osnet_x0_25_msmt17.pt"   # Auto-downloaded by boxmot on first run
BOTSORT_DEVICE  = "cpu"                      # "cpu" or "cuda"
BOTSORT_HALF    = False                      # FP16 — True only if CUDA available

TRACK_THRESH  = 0.5     # Minimum detection confidence to initialise a track
TRACK_BUFFER  = 50      # Frames to keep lost track alive (10s at 5fps)
MATCH_THRESH  = 0.8     # IOU threshold for motion-based track matching
MIN_BOX_AREA  = 100     # Ignore tiny bounding boxes (pixels squared)

# ─── RTMPOSE ─────────────────────────────────────────────────────────────────
RTMPOSE_BACKEND         = "onnxruntime"
RTMPOSE_DEVICE          = "cpu"
KEYPOINT_CONF_THRESHOLD = 0.4

KP = {
    "nose":           0,
    "left_eye":       1,
    "right_eye":      2,
    "left_ear":       3,
    "right_ear":      4,
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":    10,
    "left_hip":       11,
    "right_hip":      12,
    "left_knee":      13,
    "right_knee":     14,
    "left_ankle":     15,
    "right_ankle":    16,
}

# ─── POSTURE CLASSIFIER ───────────────────────────────────────────────────────
SPINE_UPRIGHT_MAX             = 15
SPINE_MILD_SLOUCH_MAX         = 30
HEAD_LEVEL_MAX                = 20
SHOULDER_SLOPE_TENSE_THRESHOLD= 15   # pixels — shoulder height asymmetry

# ─── ACTIVITY CLASSIFIER ─────────────────────────────────────────────────────
MOVEMENT_IDLE_MAX   = 8
MOVEMENT_ACTIVE_MAX = 35

# Restlessness: std deviation of recent frame displacements
RESTLESSNESS_WINDOW_FRAMES  = 20     # ~4 seconds at 5fps
RESTLESSNESS_HIGH_THRESHOLD = 12.0   # std-dev pixels — flagged as restless
RESTLESSNESS_LOW_THRESHOLD  = 2.0    # std-dev pixels — very frozen

# Rapid posture shifts within a short window indicate agitation
POSTURE_SHIFT_WINDOW_FRAMES = 30     # ~6 seconds
POSTURE_SHIFT_HIGH_COUNT    = 4      # >= 4 different labels = agitated

IDLE_ALERT_SECONDS = 1800            # 30 min continuous idle = prolonged

# ─── TEMPORAL WINDOWS ────────────────────────────────────────────────────────
# One 2-hour TUMBLING window (hard reset every 2 hours).
# Inside it, four ROLLING sub-windows for trend analysis at different scales.
TUMBLING_WINDOW_SEC = 7200

ROLLING_WINDOWS = {
    "micro":  600,    # 10 min — immediate state
    "short":  1800,   # 30 min — session trend
    "medium": 3600,   # 1 hour — half-block trend
    "full":   7200,   # 2 hours — full tumbling block summary
}

# Recompute wellness indices only after this many new observations per person.
# 1 = recompute every frame. Higher values reduce CPU usage but add slight latency.
FUSION_COMPUTE_EVERY_N_OBSERVATIONS = 3

# ─── MENTAL WELLNESS INDICES ─────────────────────────────────────────────────
# Fatigue Index (0-100): higher = more fatigued
FATIGUE_WEIGHTS = {
    "prolonged_idle_rate":   0.35,
    "slouching_rate":        0.25,
    "low_activity_rate":     0.20,
    "posture_rigidity_rate": 0.20,
}

# Stress / Agitation Index (0-100): higher = more stressed
STRESS_WEIGHTS = {
    "restlessness_rate":     0.35,
    "shoulder_tension_rate": 0.25,
    "posture_shift_rate":    0.25,
    "rigid_upright_rate":    0.15,
}

# Engagement Index (0-100): higher = more engaged
ENGAGEMENT_WEIGHTS = {
    "upright_rate":    0.35,
    "active_rate":     0.35,
    "head_level_rate": 0.30,
}

# Label thresholds — Fatigue & Stress: higher is worse
FATIGUE_LABELS = [
    (60, "High Fatigue"),
    (30, "Mild Fatigue"),
    (0,  "Normal"),
]
STRESS_LABELS = [
    (60, "High Stress"),
    (30, "Mild Stress"),
    (0,  "Normal"),
]
# Engagement: higher is better
ENGAGEMENT_LABELS = [
    (60, "Highly Engaged"),
    (30, "Engaged"),
    (0,  "Distracted / Idle"),
]

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
OUTPUT_DIR           = "output/results"
SAVE_ANNOTATED_VIDEO = True
LOG_LEVEL            = "INFO"
