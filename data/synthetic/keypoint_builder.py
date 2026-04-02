# data/synthetic/keypoint_builder.py
# Generates realistic RTMPose-format keypoints from a posture label.
#
# The pipeline classifier works by computing angles from keypoints.
# For synthetic data to be valid, the keypoints must actually produce
# the angles that correspond to the given posture label — otherwise the
# classifier will classify them differently and the test will be meaningless.
#
# All coordinates are in pixel space for a 1280x720 frame.
# Person is assumed to be seated at a desk, camera roughly front-facing.

import math
import random


# ─── CANONICAL SEATED SKELETON ─────────────────────────────────────────────────
# Base coordinates for a seated person in the centre of a 1280x720 frame.
# Body height in frame: ~330px (typical for a person 2-3m from CCTV camera)

BASE = {
    "nose":           (640, 200),
    "left_eye":       (625, 190),
    "right_eye":      (655, 190),
    "left_ear":       (610, 195),
    "right_ear":      (670, 195),
    "left_shoulder":  (600, 270),
    "right_shoulder": (680, 270),
    "left_elbow":     (575, 340),
    "right_elbow":    (705, 340),
    "left_wrist":     (560, 405),
    "right_wrist":    (720, 405),
    "left_hip":       (615, 400),
    "right_hip":      (665, 400),
    "left_knee":      (610, 430),   # seated: knees close to hip height
    "right_knee":     (670, 430),
    "left_ankle":     (608, 510),
    "right_ankle":    (672, 510),
}

# Default confidence for all keypoints
DEFAULT_CONF = 0.88


def _add_noise(val: float, sigma: float = 3.0) -> float:
    """Adds small Gaussian noise to a coordinate value."""
    return val + random.gauss(0, sigma)


def build_keypoints(posture: str, noise_sigma: float = 3.0) -> dict:
    """
    Generates a named keypoint dict that will produce the given posture label
    when processed by PostureClassifier.

    The keypoints are geometrically correct — spine angles and head tilts
    are set to values that will trigger the correct classifier branch.

    Parameters:
        posture:     One of: upright / mild_slouch / slouching / head_down / unknown
        noise_sigma: Pixel noise added to each coordinate (simulates detection jitter).

    Returns:
        Dict matching the format from PoseEstimator:
        {"left_shoulder": {"x": float, "y": float, "conf": float}, ...}
    """
    # Start from the canonical base and apply posture-specific offsets
    coords = {k: list(v) for k, v in BASE.items()}

    if posture == "upright":
        # Spine angle ~8° — within SPINE_UPRIGHT_MAX (15°)
        # Achieved by keeping shoulders almost directly above hips
        _apply_spine_angle(coords, angle_deg=8)
        _apply_head_tilt(coords, tilt_deg=5)
        _apply_shoulder_slope(coords, slope_px=5)

    elif posture == "mild_slouch":
        # Spine angle ~22° — between SPINE_UPRIGHT_MAX and SPINE_MILD_SLOUCH_MAX
        _apply_spine_angle(coords, angle_deg=22)
        _apply_head_tilt(coords, tilt_deg=12)
        _apply_shoulder_slope(coords, slope_px=8)

    elif posture == "slouching":
        # Spine angle ~38° — above SPINE_MILD_SLOUCH_MAX (30°)
        _apply_spine_angle(coords, angle_deg=38)
        _apply_head_tilt(coords, tilt_deg=15)
        _apply_shoulder_slope(coords, slope_px=10)

    elif posture == "head_down":
        # Head tilt >HEAD_LEVEL_MAX (20°) — this overrides spine check
        _apply_spine_angle(coords, angle_deg=20)
        _apply_head_tilt(coords, tilt_deg=28)
        _apply_shoulder_slope(coords, slope_px=12)

    elif posture == "unknown":
        # Return empty dict — simulates missing/low-confidence keypoints
        return {}

    # Build named keypoint dict with noise
    keypoints = {}
    for name, (x, y) in coords.items():
        keypoints[name] = {
            "x":    _add_noise(x, noise_sigma),
            "y":    _add_noise(y, noise_sigma),
            "conf": DEFAULT_CONF - random.uniform(0, 0.1),
        }

    return keypoints


def _apply_spine_angle(coords: dict, angle_deg: float):
    """
    Shifts the entire upper body horizontally relative to the hips so that
    the computed spine angle matches angle_deg.

    The PostureClassifier computes:
        spine_angle = atan2(|dx|, |dy|)
    where dx = mid_shoulder.x - mid_hip.x
          dy = mid_shoulder.y - mid_hip.y (always negative, shoulders above hips)

    To achieve angle_deg exactly:
        dx = |dy| * tan(angle_deg)
    We shift both shoulders by the full dx amount (not split across them),
    because the classifier uses the mid-shoulder point.

    Parameters:
        coords:    Coordinate dict to modify in-place.
        angle_deg: Target spine angle from vertical in degrees.
    """
    mid_hip_y      = (coords["left_hip"][1]      + coords["right_hip"][1])      / 2
    mid_shoulder_y = (coords["left_shoulder"][1] + coords["right_shoulder"][1]) / 2

    dy = mid_shoulder_y - mid_hip_y          # negative (shoulders above hips in image)
    dx = abs(dy) * math.tan(math.radians(angle_deg))

    # Shift the entire upper body by dx (forward lean)
    upper_body = [
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "nose",          "left_eye",   "right_eye",
        "left_ear",      "right_ear",
    ]
    for kp in upper_body:
        coords[kp][0] += dx


def _apply_head_tilt(coords: dict, tilt_deg: float):
    """
    Shifts the nose forward relative to mid-shoulder to produce a head tilt angle.

    PostureClassifier computes:
        head_tilt = atan2(|dx|, |dy|)
    where dx = nose.x - mid_shoulder.x
          dy = nose.y - mid_shoulder.y

    Parameters:
        coords:   Coordinate dict to modify in-place.
        tilt_deg: Target head tilt angle from vertical in degrees.
    """
    mid_s_x = (coords["left_shoulder"][0] + coords["right_shoulder"][0]) / 2
    mid_s_y = (coords["left_shoulder"][1] + coords["right_shoulder"][1]) / 2

    dy = coords["nose"][1] - mid_s_y          # negative (nose above shoulders)
    dx = abs(dy) * math.tan(math.radians(tilt_deg))

    # Shift nose, eyes, ears forward
    for kp in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
        coords[kp][0] += dx


def _apply_shoulder_slope(coords: dict, slope_px: float):
    """
    Creates an asymmetry in shoulder heights to simulate muscle tension.

    A positive slope_px raises the left shoulder relative to the right.
    PostureClassifier checks: abs(left_shoulder.y - right_shoulder.y) >= threshold

    Parameters:
        coords:   Coordinate dict to modify in-place.
        slope_px: Pixel height difference between left and right shoulders.
    """
    coords["left_shoulder"][1]  -= slope_px / 2
    coords["right_shoulder"][1] += slope_px / 2
