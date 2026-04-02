# src/tracking/tracker.py
# Wraps BoTSORT to assign persistent Re-ID-backed identities to persons.
#
# WHY BOTSORT over ByteTrack for this use case:
# ByteTrack is purely motion-based — it loses a track after TRACK_BUFFER frames.
# In an office, an employee may leave for a meeting or break (30-120 min) and
# return. ByteTrack would assign them a new ID, wiping their wellness history.
# BoTSORT combines motion matching WITH appearance embeddings (Re-ID), so when
# Person 3 re-enters after 90 minutes, BoTSORT recognises their appearance and
# restores their original track_id — preserving all historical wellness data.

import numpy as np
from pathlib import Path
from typing import Any, Callable, cast
import boxmot
from config.settings import (
    REID_MODEL_PATH,
    BOTSORT_DEVICE,
    BOTSORT_HALF,
    TRACK_THRESH,
    TRACK_BUFFER,
    MATCH_THRESH,
    MIN_BOX_AREA,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonTracker:
    """
    Assigns persistent identities to detected persons across frames using BoTSORT.

    BoTSORT combines:
        - Motion model (Kalman filter, like ByteTrack)
        - Appearance Re-ID embeddings (osnet model)

    The Re-ID component allows re-identification of persons who leave and
    re-enter the frame — essential for long-duration office monitoring.

    Output format per track:
        {"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}
    """

    def __init__(self):
        self.tracker: Any = None

    def load(self) -> bool:
        """
        Initialises the BoTSORT tracker and Re-ID model.
        boxmot will auto-download the Re-ID weights on first run.

        Returns:
            True if initialised successfully, False otherwise.
        """
        try:
            botsort_ctor = cast(Callable[..., Any], getattr(boxmot, "BoTSORT"))
            self.tracker = botsort_ctor(
                reid_weights=Path(REID_MODEL_PATH),
                device=BOTSORT_DEVICE,
                half=BOTSORT_HALF,
                track_high_thresh=TRACK_THRESH,
                track_buffer=TRACK_BUFFER,
                match_thresh=MATCH_THRESH,
                min_box_area=MIN_BOX_AREA,
            )
            logger.info(
                f"BoTSORT initialised | device={BOTSORT_DEVICE} | "
                f"reid_weights={REID_MODEL_PATH}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialise BoTSORT: {e}")
            return False

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list:
        """
        Updates tracker with new detections and returns tracked persons.

        Parameters:
            detections: numpy array shape (N, 5) — [x1, y1, x2, y2, conf]
                        from PersonDetector.detect()
            frame:      Current preprocessed BGR frame — required by BoTSORT
                        to extract Re-ID appearance embeddings from each bbox.

        Returns:
            List of dicts:
            [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        if self.tracker is None:
            logger.error("Tracker not loaded. Call load() first.")
            return []

        try:
            # boxmot expects (N, 6): [x1,y1,x2,y2,conf,class_id]
            if detections is None:
                detections = np.empty((0, 5))

            class_ids  = np.zeros((len(detections), 1))
            dets_input = np.hstack([detections, class_ids])

            tracks = self.tracker.update(dets_input, frame)

            if tracks is None or len(tracks) == 0:
                return []

            # BoTSORT output columns: [x1,y1,x2,y2,track_id,conf,class_id,...]
            tracked_persons = []
            for t in tracks:
                tracked_persons.append({
                    "track_id":   int(t[4]),
                    "bbox":       [float(t[0]), float(t[1]),
                                   float(t[2]), float(t[3])],
                    "confidence": float(t[5]),
                })

            logger.debug(f"BoTSORT tracking {len(tracked_persons)} person(s)")
            return tracked_persons

        except Exception as e:
            logger.error(f"BoTSORT update error: {e}")
            return []
