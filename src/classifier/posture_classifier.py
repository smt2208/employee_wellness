# src/classifier/posture_classifier.py
# Rule-based classifier that converts RTMPose keypoints into signals
# that directly feed the three mental wellness indices:
#   Fatigue Index, Stress/Agitation Index, Engagement Index.
#
# Every signal returned by this classifier maps to a specific index.
# Nothing is computed here for its own sake — each output has a purpose.

import math
import statistics
from collections import deque
from config.settings import (
    SPINE_UPRIGHT_MAX,
    SPINE_MILD_SLOUCH_MAX,
    HEAD_LEVEL_MAX,
    SHOULDER_SLOPE_TENSE_THRESHOLD,
    MOVEMENT_IDLE_MAX,
    MOVEMENT_ACTIVE_MAX,
    RESTLESSNESS_WINDOW_FRAMES,
    RESTLESSNESS_HIGH_THRESHOLD,
    RESTLESSNESS_LOW_THRESHOLD,
    POSTURE_SHIFT_WINDOW_FRAMES,
    POSTURE_SHIFT_HIGH_COUNT,
    IDLE_ALERT_SECONDS,
    TARGET_FPS,
)
from utils.logger import get_logger

logger = get_logger(__name__)

MOVEMENT_HISTORY_FRAMES = max(RESTLESSNESS_WINDOW_FRAMES, POSTURE_SHIFT_WINDOW_FRAMES)


class PostureClassifier:
    """
    Converts RTMPose keypoints into posture, activity, and mental wellness signals.

    Output per person (added to their dict):
        posture          : str  — upright / mild_slouch / slouching / head_down / unknown
        activity         : str  — walking / standing / sitting_active / sitting_idle /
                                   prolonged_idle / unknown
        shoulder_tension : bool — elevated/uneven shoulders (stress signal)
        restless         : bool — high movement variance (stress/agitation signal)
        posture_shift    : bool — rapid label changes in recent window (stress signal)
        frozen           : bool — near-zero movement for extended time (fatigue signal)
        head_level       : bool — head not drooping (engagement signal)
    """

    def __init__(self):
        # Per-person movement displacement history {track_id: deque of floats}
        self._displacement_history: dict = {}
        # Per-person posture label history {track_id: deque of str}
        self._posture_history: dict = {}
        # Per-person bbox centre history {track_id: deque(maxlen=2) of (cx,cy)}
        self._centre_history: dict = {}
        # Per-person idle frame counter {track_id: int}
        self._idle_counters: dict = {}

    def classify(self, person: dict) -> dict:
        """
        Classifies all posture and wellness signals for a single person.

        Parameters:
            person: Dict with 'track_id', 'bbox', 'keypoints'.

        Returns:
            Input dict extended with all signal fields.
        """
        tid       = person["track_id"]
        kps       = person.get("keypoints", {})
        bbox      = person["bbox"]
        result    = person.copy()

        # ── Geometry-based signals ────────────────────────────────────────────
        posture          = self._classify_posture(kps)
        head_level       = self._is_head_level(kps)
        shoulder_tension = self._has_shoulder_tension(kps)

        # ── Movement-based signals ────────────────────────────────────────────
        centre      = self._bbox_centre(bbox)
        displacement= self._update_displacement(tid, centre)
        activity    = self._classify_activity(tid, displacement, kps)
        restless    = self._is_restless(tid)
        frozen      = self._is_frozen(tid)

        # ── Posture shift signal (agitation) ──────────────────────────────────
        posture_shift = self._update_posture_history(tid, posture)

        result.update({
            "posture":          posture,
            "activity":         activity,
            "shoulder_tension": shoulder_tension,
            "restless":         restless,
            "posture_shift":    posture_shift,
            "frozen":           frozen,
            "head_level":       head_level,
        })

        logger.debug(
            f"Track {tid} | posture={posture} | activity={activity} | "
            f"tension={shoulder_tension} | restless={restless} | "
            f"frozen={frozen} | head_level={head_level}"
        )
        return result

    # ─── POSTURE ──────────────────────────────────────────────────────────────

    def _classify_posture(self, kps: dict) -> str:
        """
        Determines posture label from spine and head geometry.

        Priority: head_down > slouching > mild_slouch > upright
        Head-down is highest priority because it is the strongest
        single fatigue signal we can observe from CCTV.

        Parameters:
            kps: Named keypoint dict from PoseEstimator.

        Returns:
            Posture label string.
        """
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in kps for k in required):
            return "unknown"

        head_tilt   = self._compute_head_tilt(kps)
        spine_angle = self._compute_spine_angle(kps)

        # Head-down overrides spine check
        if head_tilt is not None and head_tilt > HEAD_LEVEL_MAX:
            return "head_down"

        if spine_angle <= SPINE_UPRIGHT_MAX:
            return "upright"
        elif spine_angle <= SPINE_MILD_SLOUCH_MAX:
            return "mild_slouch"
        else:
            return "slouching"

    def _compute_spine_angle(self, kps: dict) -> float:
        """
        Angle of spine vector (mid_hip → mid_shoulder) from vertical.
        0° = perfectly upright. Increases as person leans forward.
        """
        ls = kps["left_shoulder"]
        rs = kps["right_shoulder"]
        lh = kps["left_hip"]
        rh = kps["right_hip"]

        mid_s = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
        mid_h = ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)

        dx = mid_s[0] - mid_h[0]
        dy = mid_s[1] - mid_h[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def _compute_head_tilt(self, kps: dict):
        """
        Angle of (mid_shoulder → nose) from vertical.
        Returns None if nose keypoint is missing.
        """
        if "nose" not in kps:
            return None
        ls   = kps["left_shoulder"]
        rs   = kps["right_shoulder"]
        nose = kps["nose"]
        mid_s = ((ls["x"] + rs["x"]) / 2, (ls["y"] + rs["y"]) / 2)
        dx = nose["x"] - mid_s[0]
        dy = nose["y"] - mid_s[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    def _is_head_level(self, kps: dict) -> bool:
        """
        Returns True if head tilt is within normal range (head not drooping).
        Used as a positive engagement signal.
        """
        tilt = self._compute_head_tilt(kps)
        if tilt is None:
            return True   # Can't determine — default to no penalty
        return tilt <= HEAD_LEVEL_MAX

    def _has_shoulder_tension(self, kps: dict) -> bool:
        """
        Returns True if shoulder height asymmetry exceeds the tension threshold.
        Uneven shoulders indicate muscle tension — a key stress signal.
        """
        if "left_shoulder" not in kps or "right_shoulder" not in kps:
            return False
        slope = abs(kps["left_shoulder"]["y"] - kps["right_shoulder"]["y"])
        return slope >= SHOULDER_SLOPE_TENSE_THRESHOLD

    # ─── ACTIVITY ─────────────────────────────────────────────────────────────

    def _classify_activity(
        self, tid: int, displacement: float, kps: dict
    ) -> str:
        """
        Determines activity label from bbox displacement and lower-body geometry.

        Parameters:
            tid:          Track ID.
            displacement: Pixel distance moved since last frame.
            kps:          Named keypoint dict.

        Returns:
            Activity label string.
        """
        # Walking: large displacement
        if displacement > MOVEMENT_ACTIVE_MAX:
            self._idle_counters[tid] = 0
            return "walking"

        is_sitting = self._is_sitting(kps)

        if displacement < MOVEMENT_IDLE_MAX:
            self._idle_counters[tid] = self._idle_counters.get(tid, 0) + 1
            idle_seconds = self._idle_counters[tid] / TARGET_FPS
            if idle_seconds >= IDLE_ALERT_SECONDS:
                return "prolonged_idle"
            return "sitting_idle" if is_sitting else "standing"

        # Active range (8-35px)
        self._idle_counters[tid] = 0
        return "sitting_active" if is_sitting else "standing"

    def _is_sitting(self, kps: dict) -> bool:
        """
        Estimates whether person is sitting by comparing hip-knee vertical gap.
        Sitting: knees close to hip height. Standing: knees well below hips.
        """
        has_hip   = "left_hip"  in kps or "right_hip"  in kps
        has_knee  = "left_knee" in kps or "right_knee" in kps
        if not (has_hip and has_knee):
            return False

        hip = kps.get("left_hip") or kps.get("right_hip")
        knee = kps.get("left_knee") or kps.get("right_knee")
        if not hip or not knee:
            return False

        hip_y = hip["y"]
        knee_y = knee["y"]
        return abs(knee_y - hip_y) < 30   # pixel threshold for seated geometry

    # ─── MOVEMENT SIGNALS ─────────────────────────────────────────────────────

    def _bbox_centre(self, bbox: list) -> tuple:
        """Returns the (cx, cy) centre of a bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _update_displacement(self, tid: int, centre: tuple) -> float:
        """
        Computes pixel displacement from last frame centre and updates history.
        Also stores the displacement value for restlessness computation.

        Returns:
            Displacement in pixels (0.0 if first frame for this track).
        """
        if tid not in self._centre_history:
            self._centre_history[tid]      = deque(maxlen=2)
            self._displacement_history[tid]= deque(maxlen=MOVEMENT_HISTORY_FRAMES)

        history = self._centre_history[tid]
        history.append(centre)

        if len(history) < 2:
            self._displacement_history[tid].append(0.0)
            return 0.0

        prev  = history[-2]
        dx    = centre[0] - prev[0]
        dy    = centre[1] - prev[1]
        disp  = math.sqrt(dx**2 + dy**2)
        self._displacement_history[tid].append(disp)
        return disp

    def _is_restless(self, tid: int) -> bool:
        """
        Returns True if recent movement variance is high (fidgeting/agitation).
        Uses standard deviation of displacement history over RESTLESSNESS_WINDOW_FRAMES.
        High std-dev = erratic movement = stress signal.
        """
        history = self._displacement_history.get(tid)
        if not history or len(history) < RESTLESSNESS_WINDOW_FRAMES // 2:
            return False
        recent = list(history)[-RESTLESSNESS_WINDOW_FRAMES:]
        if len(recent) < 2:
            return False
        std = statistics.stdev(recent)
        return std >= RESTLESSNESS_HIGH_THRESHOLD

    def _is_frozen(self, tid: int) -> bool:
        """
        Returns True if movement is essentially zero for the recent window.
        Frozen posture (no micro-adjustments at all) is a fatigue/rigidity signal.
        """
        history = self._displacement_history.get(tid)
        if not history or len(history) < RESTLESSNESS_WINDOW_FRAMES // 2:
            return False
        recent = list(history)[-RESTLESSNESS_WINDOW_FRAMES:]
        if len(recent) < 2:
            return False
        std = statistics.stdev(recent)
        return std <= RESTLESSNESS_LOW_THRESHOLD

    def _update_posture_history(self, tid: int, posture: str) -> bool:
        """
        Appends posture label to history and returns True if rapid shifts detected.
        Rapid posture changes (can't settle) indicate agitation — a stress signal.

        Returns:
            True if number of unique posture labels in recent window is high.
        """
        if tid not in self._posture_history:
            self._posture_history[tid] = deque(maxlen=POSTURE_SHIFT_WINDOW_FRAMES)

        self._posture_history[tid].append(posture)
        recent = list(self._posture_history[tid])[-POSTURE_SHIFT_WINDOW_FRAMES:]
        unique_count = len(set(recent))
        return unique_count >= POSTURE_SHIFT_HIGH_COUNT

    def reset_person(self, tid: int):
        """Clears all stored history for a lost track."""
        self._displacement_history.pop(tid, None)
        self._posture_history.pop(tid, None)
        self._centre_history.pop(tid, None)
        self._idle_counters.pop(tid, None)
