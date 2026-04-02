# data/synthetic/profiles.py
# Defines employee behaviour profiles used by the synthetic data generator.
#
# Each profile represents a distinct mental wellness scenario that the
# pipeline should be able to detect and classify correctly.
# These are the "ground truth" inputs — if the pipeline is working, the
# indices it produces should match the expected state described here.
#
# Profile structure:
#   Each profile is a dict of signal RATES (0.0 to 1.0) representing
#   the fraction of frames where each signal is True or each label occurs.
#   Small random noise is added during generation to make data realistic.

PROFILES = {

    # ─── PROFILE A: Healthy Engaged Employee ──────────────────────────────────
    # Expected output: Normal Fatigue, Normal Stress, Highly Engaged
    "healthy_engaged": {
        "description": (
            "Employee working productively. Sitting upright, "
            "regular small movements, calm, head level."
        ),
        "expected": {
            "fatigue":    "Normal",
            "stress":     "Normal",
            "engagement": "Highly Engaged",
        },
        # Posture label probabilities (must sum to ~1.0)
        "posture_probs": {
            "upright":     0.70,
            "mild_slouch": 0.20,
            "slouching":   0.05,
            "head_down":   0.03,
            "unknown":     0.02,
        },
        # Activity label probabilities
        "activity_probs": {
            "sitting_active": 0.50,
            "sitting_idle":   0.25,
            "standing":       0.15,
            "walking":        0.08,
            "prolonged_idle": 0.02,
        },
        # Signal rates (fraction of frames where signal is True)
        "shoulder_tension_rate": 0.05,
        "restless_rate":         0.04,
        "posture_shift_rate":    0.05,
        "frozen_rate":           0.06,
        "head_level_rate":       0.90,
        # Movement: displacement per frame in pixels (mean, std)
        "displacement": (5.0, 3.0),
    },

    # ─── PROFILE B: Fatigued Employee ─────────────────────────────────────────
    # Expected output: High Fatigue, Normal Stress, Distracted / Idle
    "fatigued": {
        "description": (
            "Employee showing signs of fatigue. Sustained slouching, "
            "head dropping, long idle periods, minimal movement."
        ),
        "expected": {
            "fatigue":    "High Fatigue",
            "stress":     "Normal",
            "engagement": "Distracted / Idle",
        },
        "posture_probs": {
            "upright":     0.05,
            "mild_slouch": 0.25,
            "slouching":   0.40,
            "head_down":   0.25,
            "unknown":     0.05,
        },
        "activity_probs": {
            "sitting_active": 0.05,
            "sitting_idle":   0.30,
            "standing":       0.05,
            "walking":        0.02,
            "prolonged_idle": 0.58,
        },
        "shoulder_tension_rate": 0.10,
        "restless_rate":         0.05,
        "posture_shift_rate":    0.06,
        "frozen_rate":           0.70,
        "head_level_rate":       0.20,
        "displacement": (1.5, 1.0),
    },

    # ─── PROFILE C: Stressed / Agitated Employee ──────────────────────────────
    # Expected output: Normal Fatigue, High Stress, Engaged
    "stressed": {
        "description": (
            "Employee under stress. Restless movements, frequent posture "
            "changes, shoulder tension, can't settle."
        ),
        "expected": {
            "fatigue":    "Normal",
            "stress":     "High Stress",
            "engagement": "Engaged",
        },
        "posture_probs": {
            "upright":     0.30,
            "mild_slouch": 0.30,
            "slouching":   0.20,
            "head_down":   0.10,
            "unknown":     0.10,
        },
        "activity_probs": {
            "sitting_active": 0.40,
            "sitting_idle":   0.20,
            "standing":       0.20,
            "walking":        0.15,
            "prolonged_idle": 0.05,
        },
        "shoulder_tension_rate": 0.65,
        "restless_rate":         0.70,
        "posture_shift_rate":    0.65,
        "frozen_rate":           0.05,
        "head_level_rate":       0.55,
        "displacement": (18.0, 12.0),   # high variance = restless
    },

    # ─── PROFILE D: Disengaged but Not Fatigued ───────────────────────────────
    # Expected output: Mild Fatigue, Normal Stress, Distracted / Idle
    "disengaged": {
        "description": (
            "Employee present but not engaged. Mild slouching, "
            "sitting idle frequently, low energy but not visibly stressed."
        ),
        "expected": {
            "fatigue":    "Mild Fatigue",
            "stress":     "Normal",
            "engagement": "Distracted / Idle",
        },
        "posture_probs": {
            "upright":     0.10,
            "mild_slouch": 0.45,
            "slouching":   0.30,
            "head_down":   0.10,
            "unknown":     0.05,
        },
        "activity_probs": {
            "sitting_active": 0.10,
            "sitting_idle":   0.55,
            "standing":       0.10,
            "walking":        0.05,
            "prolonged_idle": 0.20,
        },
        "shoulder_tension_rate": 0.12,
        "restless_rate":         0.08,
        "posture_shift_rate":    0.10,
        "frozen_rate":           0.35,
        "head_level_rate":       0.35,
        "displacement": (3.0, 2.0),
    },
}
