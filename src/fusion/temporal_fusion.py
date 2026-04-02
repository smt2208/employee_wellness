# src/fusion/temporal_fusion.py
# Computes the three mental wellness indices using a tumbling + rolling
# window architecture.
#
# WINDOW STRUCTURE:
#   Tumbling window (2 hours) — hard resets every 2 hours.
#     Matches real office work blocks (morning / afternoon / late).
#     At the end of each block, a sealed session report is produced.
#
#   Rolling sub-windows inside each tumbling block:
#     micro  (10 min) — immediate state, catches quick changes
#     short  (30 min) — session-level trend
#     medium (1 hour) — half-block trend
#     full   (2 hour) — the whole tumbling window
#
# THREE INDICES (all 0-100):
#   Fatigue Index    — higher = more fatigued
#   Stress Index     — higher = more stressed / agitated
#   Engagement Index — higher = more engaged (good)

import time
from collections import deque
from typing import Optional
from config.settings import (
    TUMBLING_WINDOW_SEC,
    ROLLING_WINDOWS,
    FUSION_COMPUTE_EVERY_N_OBSERVATIONS,
    FATIGUE_WEIGHTS,
    STRESS_WEIGHTS,
    ENGAGEMENT_WEIGHTS,
    FATIGUE_LABELS,
    STRESS_LABELS,
    ENGAGEMENT_LABELS,
    TARGET_FPS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _score_to_label(score: float, label_table: list) -> str:
    """
    Converts a 0-100 index score to a human-readable label.

    Parameters:
        score:       Float 0-100.
        label_table: List of (threshold, label) pairs, descending order.

    Returns:
        Label string.
    """
    for threshold, label in label_table:
        if score >= threshold:
            return label
    return label_table[-1][1]


class Observation:
    """
    Stores one frame's worth of binary signals for a single person.
    Each field maps directly to a component of one of the three indices.
    """
    __slots__ = [
        "timestamp",
        # Fatigue signals
        "prolonged_idle",
        "slouching",
        "low_activity",
        "frozen",
        # Stress signals
        "restless",
        "shoulder_tension",
        "posture_shift",
        "rigid_upright",
        # Engagement signals
        "upright",
        "active",
        "head_level",
    ]

    def __init__(self, timestamp: float, person: dict):
        """
        Extracts binary signals from a classified person dict.

        Parameters:
            timestamp: Monotonic timestamp of this frame.
            person:    Dict from PostureClassifier with all signal fields.
        """
        self.timestamp = timestamp

        posture  = person.get("posture",  "unknown")
        activity = person.get("activity", "unknown")

        # ── Fatigue signals ───────────────────────────────────────────────────
        self.prolonged_idle = activity == "prolonged_idle"
        self.slouching      = posture in ("slouching", "head_down")
        # Keep prolonged_idle distinct so fatigue is not double-counted.
        self.low_activity   = activity == "sitting_idle"
        self.frozen         = person.get("frozen", False)

        # ── Stress / Agitation signals ────────────────────────────────────────
        self.restless        = person.get("restless",         False)
        self.shoulder_tension= person.get("shoulder_tension", False)
        self.posture_shift   = person.get("posture_shift",    False)
        # Rigid upright = upright but also frozen (tense, not naturally upright)
        self.rigid_upright   = (posture == "upright") and person.get("frozen", False)

        # ── Engagement signals ────────────────────────────────────────────────
        self.upright    = posture == "upright"
        self.active     = activity in ("walking", "standing", "sitting_active")
        self.head_level = person.get("head_level", True)


class PersonWellnessTracker:
    """
    Maintains the observation buffer and computes wellness indices
    for a single tracked person across the tumbling + rolling window structure.
    """

    def __init__(self, track_id: int):
        """
        Parameters:
            track_id: BoTSORT persistent ID for this person.
        """
        self.track_id = track_id

        # Max buffer = full tumbling window worth of frames
        max_frames = int(TUMBLING_WINDOW_SEC * TARGET_FPS)
        self._buffer: deque = deque(maxlen=max_frames)

        # Monotonic start for all duration math.
        self._window_start_monotonic: float = time.monotonic()
        # Wall-clock start retained for reporting.
        self._window_start_wall: float = time.time()
        self._last_seen_monotonic: float = self._window_start_monotonic

        # Completed tumbling window summaries (for longitudinal trend)
        self._completed_blocks: list = []

        # Cache to avoid recomputing all rolling windows on every frame.
        self._cached_indices: Optional[dict] = None
        self._records_since_compute: int = 0
        self._compute_every_n = max(1, int(FUSION_COMPUTE_EVERY_N_OBSERVATIONS))

    def record(self, person: dict):
        """
        Records one frame observation.

        Parameters:
            person: Classified person dict from PostureClassifier.
        """
        now_mono = time.monotonic()
        now_wall = time.time()

        # Check expiry BEFORE appending current frame so boundary frames
        # start the new tumbling block, not the old one.
        if now_mono - self._window_start_monotonic >= TUMBLING_WINDOW_SEC:
            self._seal_tumbling_window(
                sealed_at_wall=now_wall,
                next_window_start_monotonic=now_mono,
                next_window_start_wall=now_wall,
            )

        obs = Observation(now_mono, person)
        self._buffer.append(obs)
        self._last_seen_monotonic = now_mono
        self._records_since_compute += 1

    def get_indices(self, force_recompute: bool = False) -> dict:
        """
        Computes all three mental wellness indices across all rolling sub-windows.

        Parameters:
            force_recompute: If True, bypasses cache and computes fresh indices.

        Returns:
            Dict with structure:
            {
                "track_id": int,
                "windows": {
                    "micro":  {"fatigue": float, "stress": float, "engagement": float,
                               "fatigue_label": str, "stress_label": str,
                               "engagement_label": str},
                    "short":  {...},
                    "medium": {...},
                    "full":   {...},
                },
                "completed_blocks": int,
            }
        """
        if (
            not force_recompute
            and self._cached_indices is not None
            and self._records_since_compute < self._compute_every_n
        ):
            return self._cached_indices

        now     = time.monotonic()
        windows = {}

        for window_name, window_sec in ROLLING_WINDOWS.items():
            # Clamp rolling window to tumbling window boundary
            effective_sec = min(window_sec, TUMBLING_WINDOW_SEC)
            cutoff        = max(now - effective_sec, self._window_start_monotonic)

            # Aggregate directly from the deque without list allocation.
            n, counts = self._aggregate_counts_since(cutoff)

            if n == 0:
                windows[window_name] = self._empty_window()
                continue

            fatigue, stress, engagement = self._compute_scores_from_counts(counts, n)

            windows[window_name] = {
                "fatigue":          round(fatigue,    1),
                "stress":           round(stress,     1),
                "engagement":       round(engagement, 1),
                "fatigue_label":    _score_to_label(fatigue,    FATIGUE_LABELS),
                "stress_label":     _score_to_label(stress,     STRESS_LABELS),
                "engagement_label": _score_to_label(engagement, ENGAGEMENT_LABELS),
                "sample_count":     n,
            }

        result = {
            "track_id":        self.track_id,
            "windows":         windows,
            "completed_blocks":len(self._completed_blocks),
        }
        self._cached_indices = result
        self._records_since_compute = 0
        return result

    # ─── INDEX COMPUTATION ────────────────────────────────────────────────────

    def _aggregate_counts_since(self, cutoff: float):
        """
        Aggregates signal counts from newest to oldest until cutoff is crossed.
        The deque is time-ordered, so reverse iteration enables early break.
        """
        counts = {
            "prolonged_idle": 0,
            "slouching": 0,
            "low_activity": 0,
            "frozen": 0,
            "restless": 0,
            "shoulder_tension": 0,
            "posture_shift": 0,
            "rigid_upright": 0,
            "upright": 0,
            "active": 0,
            "head_level": 0,
        }
        n = 0

        for obs in reversed(self._buffer):
            if obs.timestamp < cutoff:
                break

            n += 1
            counts["prolonged_idle"] += int(obs.prolonged_idle)
            counts["slouching"] += int(obs.slouching)
            counts["low_activity"] += int(obs.low_activity)
            counts["frozen"] += int(obs.frozen)
            counts["restless"] += int(obs.restless)
            counts["shoulder_tension"] += int(obs.shoulder_tension)
            counts["posture_shift"] += int(obs.posture_shift)
            counts["rigid_upright"] += int(obs.rigid_upright)
            counts["upright"] += int(obs.upright)
            counts["active"] += int(obs.active)
            counts["head_level"] += int(obs.head_level)

        return n, counts

    def _compute_scores_from_counts(self, counts: dict, n: int):
        """
        Computes all three wellness scores from aggregated signal counts.
        """
        inv_n = 1.0 / n

        fatigue = 100 * (
            FATIGUE_WEIGHTS["prolonged_idle_rate"]   * (counts["prolonged_idle"] * inv_n)
            + FATIGUE_WEIGHTS["slouching_rate"]      * (counts["slouching"] * inv_n)
            + FATIGUE_WEIGHTS["low_activity_rate"]   * (counts["low_activity"] * inv_n)
            + FATIGUE_WEIGHTS["posture_rigidity_rate"] * (counts["frozen"] * inv_n)
        )

        stress = 100 * (
            STRESS_WEIGHTS["restlessness_rate"]      * (counts["restless"] * inv_n)
            + STRESS_WEIGHTS["shoulder_tension_rate"] * (counts["shoulder_tension"] * inv_n)
            + STRESS_WEIGHTS["posture_shift_rate"]  * (counts["posture_shift"] * inv_n)
            + STRESS_WEIGHTS["rigid_upright_rate"]  * (counts["rigid_upright"] * inv_n)
        )

        engagement = 100 * (
            ENGAGEMENT_WEIGHTS["upright_rate"]       * (counts["upright"] * inv_n)
            + ENGAGEMENT_WEIGHTS["active_rate"]     * (counts["active"] * inv_n)
            + ENGAGEMENT_WEIGHTS["head_level_rate"] * (counts["head_level"] * inv_n)
        )

        return fatigue, stress, engagement

    # ─── TUMBLING WINDOW MANAGEMENT ───────────────────────────────────────────

    def _seal_tumbling_window(
        self,
        sealed_at_wall: Optional[float] = None,
        next_window_start_monotonic: Optional[float] = None,
        next_window_start_wall: Optional[float] = None,
    ):
        """
        Seals the current tumbling window by computing a final summary,
        stores it in completed_blocks, then resets the buffer for the next block.
        """
        if sealed_at_wall is None:
            sealed_at_wall = time.time()

        n = len(self._buffer)
        if n > 0:
            _, counts = self._aggregate_counts_since(self._window_start_monotonic)
            fatigue, stress, engagement = self._compute_scores_from_counts(counts, n)
            block_summary = {
                "block_number": len(self._completed_blocks) + 1,
                "start_time":   self._window_start_wall,
                "end_time":     sealed_at_wall,
                "fatigue":      round(fatigue,    1),
                "stress":       round(stress,     1),
                "engagement":   round(engagement, 1),
                "sample_count": n,
            }
            self._completed_blocks.append(block_summary)
            logger.info(
                f"Track {self.track_id} | 2-hour block sealed | "
                f"Fatigue={block_summary['fatigue']} | "
                f"Stress={block_summary['stress']} | "
                f"Engagement={block_summary['engagement']}"
            )

        # Reset for next tumbling window
        self._buffer.clear()

        if next_window_start_monotonic is None:
            next_window_start_monotonic = time.monotonic()
        if next_window_start_wall is None:
            next_window_start_wall = time.time()

        self._window_start_monotonic = next_window_start_monotonic
        self._window_start_wall = next_window_start_wall
        self._cached_indices = None
        self._records_since_compute = 0

    @staticmethod
    def _empty_window() -> dict:
        """Returns a placeholder dict when no observations are available yet."""
        return {
            "fatigue":          None,
            "stress":           None,
            "engagement":       None,
            "fatigue_label":    "Collecting data...",
            "stress_label":     "Collecting data...",
            "engagement_label": "Collecting data...",
            "sample_count":     0,
        }

    def seconds_since_last_seen(self, now_monotonic: Optional[float] = None) -> float:
        """Returns seconds since this person was last observed."""
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        return max(0.0, now_monotonic - self._last_seen_monotonic)

    def get_completed_blocks(self) -> list:
        """Returns list of all sealed 2-hour block summaries."""
        return self._completed_blocks


class TemporalFusion:
    """
    Session-level manager for all PersonWellnessTracker instances.

    One tracker per person — created on first appearance, preserved
    across re-entries (BoTSORT restores the same track_id on re-entry).
    """

    def __init__(self):
        self._trackers: dict = {}

    def update(self, classified_persons: list) -> list:
        """
        Records observations for all visible persons and returns their indices.

        Parameters:
            classified_persons: List from PostureClassifier with all signals.

        Returns:
            List of wellness index dicts — one per person.
        """
        results    = []
        active_ids = set()

        for person in classified_persons:
            tid = person["track_id"]
            active_ids.add(tid)

            if tid not in self._trackers:
                self._trackers[tid] = PersonWellnessTracker(tid)
                logger.info(f"Wellness tracker created for track_id={tid}")

            self._trackers[tid].record(person)
            results.append(self._trackers[tid].get_indices())

        return results

    def remove_lost_tracks(self, active_track_ids: set) -> list:
        """
        Removes trackers for persons no longer in the scene.
        Note: do NOT remove aggressively — a person may return after meetings,
        breaks, or short occlusions.

        Parameters:
            active_track_ids: Set of currently active track IDs.

        Returns:
            List of track IDs that were finally removed as stale.
        """
        removed_ids = []
        now_mono = time.monotonic()

        # Keep absent tracks for one full tumbling block before cleanup.
        max_absence_sec = TUMBLING_WINDOW_SEC

        for tid, tracker in list(self._trackers.items()):
            if tid in active_track_ids:
                continue

            if tracker.seconds_since_last_seen(now_mono) < max_absence_sec:
                continue

            logger.info(f"Removing stale track_id={tid}")
            del self._trackers[tid]
            removed_ids.append(tid)

        return removed_ids

    def get_all_indices(self) -> list:
        """Returns current wellness indices for all tracked persons."""
        return [t.get_indices(force_recompute=True) for t in self._trackers.values()]

    def get_session_summary(self) -> list:
        """
        Returns completed 2-hour block summaries for all persons.
        Used for final report at pipeline shutdown.
        """
        summary = []
        for tid, tracker in self._trackers.items():
            blocks = tracker.get_completed_blocks()
            summary.append({"track_id": tid, "blocks": blocks})
        return summary
