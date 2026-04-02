# src/pipeline.py
# Main orchestrator. Connects all pipeline stages in order.
# This is the only file that imports from all stages.
# Individual stage modules never import each other.

import cv2
import json
import os
import time
from datetime import datetime

from src.preprocessing.video_reader    import VideoReader
from src.detection.person_detector     import PersonDetector
from src.tracking.tracker              import PersonTracker
from src.pose.pose_estimator           import PoseEstimator
from src.classifier.posture_classifier import PostureClassifier
from src.fusion.temporal_fusion        import TemporalFusion
from config import validate_configuration
from config.settings                   import OUTPUT_DIR, SAVE_ANNOTATED_VIDEO, TARGET_FPS, STRESS_LABELS
from utils.logger                      import get_logger

logger = get_logger(__name__)


class WellnessPipeline:
    """
    Orchestrates all stages of the employee wellness monitoring pipeline.

    Stage order per frame:
        1. VideoReader     → preprocessed frame
        2. PersonDetector  → bounding boxes
        3. PersonTracker   → boxes + persistent IDs (BoTSORT with Re-ID)
        4. PoseEstimator   → 17 named keypoints per person
        5. PostureClassifier → posture, activity, and all mental wellness signals
        6. TemporalFusion  → Fatigue, Stress, Engagement indices per rolling window
        7. Annotation + logging
    """

    def __init__(self, video_source):
        """
        Parameters:
            video_source: File path string or int (0 = webcam).
        """
        self.video_source = video_source
        self.reader       = VideoReader(video_source)
        self.detector     = PersonDetector()
        self.tracker      = PersonTracker()
        self.estimator    = PoseEstimator()
        self.classifier   = PostureClassifier()
        self.fusion       = TemporalFusion()

        self.writer      = None
        self.session_log = []
        self.run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialise(self) -> bool:
        """
        Loads all models and opens the video source.

        Returns:
            True if all stages ready, False on any failure.
        """
        logger.info("=== Initialising Wellness Pipeline ===")

        try:
            validate_configuration()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

        if not self.reader.open():
            return False
        if not self.detector.load():
            return False
        if not self.tracker.load():
            return False
        if not self.estimator.load():
            return False

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info("All stages ready.")
        return True

    def run(self):
        """
        Runs the pipeline loop until the video ends or the user interrupts.
        """
        logger.info("=== Pipeline Running ===")
        frame_count = 0
        start_time  = time.time()

        try:
            for frame_idx, frame in self.reader.read_frames():

                detections  = self.detector.detect(frame)
                tracked     = self.tracker.update(detections, frame)
                with_pose   = self.estimator.estimate(frame, tracked)
                classified  = [self.classifier.classify(p) for p in with_pose]

                active_ids    = {p["track_id"] for p in classified}
                wellness_data = self.fusion.update(classified)
                removed_ids   = self.fusion.remove_lost_tracks(active_ids)
                for tid in removed_ids:
                    self.classifier.reset_person(tid)

                self._log_frame(frame_idx, classified, wellness_data)

                if SAVE_ANNOTATED_VIDEO:
                    annotated = self._annotate(frame, classified, wellness_data)
                    self._write_frame(annotated, frame)

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Processed {frame_count} frame(s) | Source frame {frame_idx} | "
                        f"{frame_count/elapsed:.1f} fps | Active persons: {len(classified)}"
                    )
        finally:
            self._finalise()

    # ─── LOGGING ──────────────────────────────────────────────────────────────

    def _log_frame(self, frame_idx: int, classified: list, wellness: list):
        """
        Stores per-frame results in session_log for JSON export.

        Parameters:
            frame_idx:  Current frame index.
            classified: Classified person dicts.
            wellness:   Wellness index dicts from TemporalFusion.
        """
        wellness_by_id = {w["track_id"]: w for w in wellness}

        entry = {
            "frame_idx": frame_idx,
            "timestamp": datetime.now().isoformat(),
            "persons":   [],
        }

        for person in classified:
            tid = person["track_id"]
            w   = wellness_by_id.get(tid, {})
            # Use the 10-min (micro) window for per-frame logging
            micro = w.get("windows", {}).get("micro", {})

            entry["persons"].append({
                "track_id":        tid,
                "posture":         person.get("posture",          "unknown"),
                "activity":        person.get("activity",         "unknown"),
                "shoulder_tension":person.get("shoulder_tension", False),
                "restless":        person.get("restless",         False),
                "frozen":          person.get("frozen",           False),
                "fatigue":         micro.get("fatigue"),
                "fatigue_label":   micro.get("fatigue_label"),
                "stress":          micro.get("stress"),
                "stress_label":    micro.get("stress_label"),
                "engagement":      micro.get("engagement"),
                "engagement_label":micro.get("engagement_label"),
            })

        self.session_log.append(entry)

    # ─── ANNOTATION ───────────────────────────────────────────────────────────

    def _annotate(self, frame, classified: list, wellness: list):
        """
        Draws bounding boxes and index scores onto the frame for visual inspection.
        Each person gets a colour-coded box and three index values.

        Parameters:
            frame:      Preprocessed BGR frame.
            classified: Classified person dicts.
            wellness:   Wellness index dicts.

        Returns:
            Annotated BGR frame.
        """
        annotated      = frame.copy()
        wellness_by_id = {w["track_id"]: w for w in wellness}

        for person in classified:
            tid  = person["track_id"]
            bbox = [int(v) for v in person["bbox"]]
            w    = wellness_by_id.get(tid, {})

            x1, y1, x2, y2 = bbox

            # Use micro window for real-time display
            micro = w.get("windows", {}).get("micro", {})
            fatigue    = micro.get("fatigue")
            stress     = micro.get("stress")
            engagement = micro.get("engagement")

            f_label = micro.get("fatigue_label",    "...")
            s_label = micro.get("stress_label",     "...")
            e_label = micro.get("engagement_label", "...")

            # Box colour driven by stress (most immediately actionable)
            box_color = self._stress_color(stress)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            lines = [
                f"ID:{tid}",
                f"Fatigue:    {fatigue:.0f}%  {f_label}"    if fatigue    is not None else "Fatigue: ...",
                f"Stress:     {stress:.0f}%  {s_label}"     if stress     is not None else "Stress: ...",
                f"Engagement: {engagement:.0f}%  {e_label}" if engagement is not None else "Engagement: ...",
                f"{person.get('posture','?')} | {person.get('activity','?')}",
            ]

            for i, line in enumerate(reversed(lines)):
                cv2.putText(
                    annotated, line,
                    (x1, y1 - 8 - (i * 17)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
                )

        return annotated

    @staticmethod
    def _stress_color(stress_score) -> tuple:
        """
        Returns a BGR colour based on the stress index value.
        Green = low stress, amber = moderate, red = high.
        """
        if stress_score is None:
            return (180, 180, 180)

        high_threshold = STRESS_LABELS[0][0] if len(STRESS_LABELS) >= 1 else 75
        moderate_threshold = STRESS_LABELS[1][0] if len(STRESS_LABELS) >= 2 else 50

        if stress_score >= high_threshold:
            return (0, 0, 220)    # Red — high stress
        elif stress_score >= moderate_threshold:
            return (0, 140, 255)  # Amber — moderate
        else:
            return (0, 200, 80)   # Green — minimal

    # ─── OUTPUT ───────────────────────────────────────────────────────────────

    def _write_frame(self, annotated, original):
        """Creates VideoWriter on first call and writes each annotated frame."""
        if self.writer is None:
            h, w = original.shape[:2]
            path   = os.path.join(OUTPUT_DIR, f"annotated_{self.run_id}.mp4")
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, TARGET_FPS, (w, h))
            logger.info(f"Saving annotated video to: {path}")
        self.writer.write(annotated)

    def _finalise(self):
        """Saves JSON log, seals final block, prints summary, releases resources."""
        log_path = os.path.join(OUTPUT_DIR, f"session_{self.run_id}.json")
        with open(log_path, "w") as f:
            json.dump(self.session_log, f, indent=2)
        logger.info(f"Session log saved: {log_path}")

        self.reader.release()
        if self.writer:
            self.writer.release()

        # Final per-person summary across all completed 2-hour blocks
        logger.info("=== Final Session Summary ===")
        for entry in self.fusion.get_session_summary():
            tid    = entry["track_id"]
            blocks = entry["blocks"]
            if not blocks:
                logger.info(f"Person {tid:>3} | No completed 2-hour blocks yet")
                continue
            last = blocks[-1]
            logger.info(
                f"Person {tid:>3} | Blocks completed: {len(blocks)} | "
                f"Last block → Fatigue: {last['fatigue']}% | "
                f"Stress: {last['stress']}% | "
                f"Engagement: {last['engagement']}%"
            )
