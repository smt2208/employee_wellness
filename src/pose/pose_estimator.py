# src/pose/pose_estimator.py
# Wraps RTMPose (via rtmlib) to extract body keypoints for each tracked person.
# We run RTMPose in top-down mode — YOLO gives us the bounding box,
# RTMPose estimates the 17 keypoints inside it.
# Since our entire wellness signal depends on pose quality, this is
# the most critical stage in the pipeline.

import numpy as np
from typing import Any, Callable, cast
from rtmlib import RTMPose
from config.settings import (
    RTMPOSE_BACKEND,
    RTMPOSE_DEVICE,
    KEYPOINT_CONF_THRESHOLD,
    KP,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PoseEstimator:
    """
    Estimates body keypoints for each detected person using RTMPose.

    RTMPose is chosen over YOLOv8-Pose and MediaPipe because:
        - Better accuracy on CCTV angles and partial occlusions
        - Designed for multi-person real-world environments
        - More robust upper-body keypoint detection which is what
          posture analysis depends on most

    Operates in top-down mode:
        Input:  bounding box from tracker
        Output: 17 (x, y, confidence) keypoints per person
    """

    def __init__(self):
        self.model: Any = None

    def load(self) -> bool:
        """
        Downloads and loads the RTMPose-m model weights.
        rtmlib handles model download automatically on first run.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            rtmpose_ctor = cast(Callable[..., Any], RTMPose)
            # Prefer RTMPose-m where the installed rtmlib version supports it.
            try:
                self.model = rtmpose_ctor(
                    pose="body",              # body keypoints (17 points)
                    backend=RTMPOSE_BACKEND,
                    device=RTMPOSE_DEVICE,
                    model_size="m",
                )
            except TypeError:
                # Backward-compatible fallback for rtmlib versions without model_size.
                self.model = rtmpose_ctor(
                    pose="body",              # body keypoints (17 points)
                    backend=RTMPOSE_BACKEND,
                    device=RTMPOSE_DEVICE,
                )
            logger.info(
                f"RTMPose loaded | backend={RTMPOSE_BACKEND} | device={RTMPOSE_DEVICE}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load RTMPose: {e}")
            return False

    def estimate(self, frame: np.ndarray, tracked_persons: list) -> list:
        """
        Runs pose estimation on all tracked persons in a frame.

        Parameters:
            frame:           Preprocessed BGR frame.
            tracked_persons: List of dicts from PersonTracker.update()
                             Each dict has 'track_id', 'bbox', 'confidence'

        Returns:
            Same list with 'keypoints' added to each person dict.
            keypoints format:
            {
                "nose":          {"x": float, "y": float, "conf": float},
                "left_shoulder": {"x": float, "y": float, "conf": float},
                ...
            }
            Persons where pose estimation fails get keypoints = {}
        """
        if self.model is None:
            logger.error("PoseEstimator not loaded. Call load() first.")
            return tracked_persons

        if not tracked_persons:
            return []

        results = [person.copy() for person in tracked_persons]
        bboxes  = [person["bbox"] for person in tracked_persons]

        try:
            keypoints_raw, scores = self.model(frame, bboxes=bboxes)
        except Exception as e:
            logger.warning(f"Pose estimation failed for current frame: {e}")
            for person in results:
                person["keypoints"] = {}
            return results

        if keypoints_raw is None or scores is None:
            for person in results:
                person["keypoints"] = {}
            return results

        for i, person in enumerate(results):
            if i >= len(keypoints_raw) or i >= len(scores):
                person["keypoints"] = {}
                continue

            kps   = keypoints_raw[i]  # (17, 2)
            confs = scores[i]         # (17,)

            named_keypoints = {}
            for name, idx in KP.items():
                conf = float(confs[idx])
                if conf >= KEYPOINT_CONF_THRESHOLD:
                    named_keypoints[name] = {
                        "x":    float(kps[idx][0]),
                        "y":    float(kps[idx][1]),
                        "conf": conf,
                    }

            person["keypoints"] = named_keypoints

        return results

    def _is_keypoint_valid(self, keypoints: dict, name: str) -> bool:
        """
        Helper to check if a specific keypoint exists and is confident enough.

        Parameters:
            keypoints: Named keypoints dict for a person.
            name:      Keypoint name string (e.g. 'left_shoulder').

        Returns:
            True if keypoint exists in the dict, False otherwise.
        """
        return name in keypoints
