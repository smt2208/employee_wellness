# src/detection/person_detector.py
# Wraps YOLOv8 to detect only "person" objects in a frame.
# Returns bounding boxes in (x1, y1, x2, y2, confidence) format
# which ByteTrack and RTMPose both expect.

import numpy as np
from ultralytics import YOLO
from config.settings import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_PERSON_CLASS_ID,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonDetector:
    """
    Detects people in a frame using a pretrained YOLOv8 model.

    Filters out all non-person detections and returns clean
    bounding boxes ready for the tracker.
    """

    def __init__(self):
        self.model = None

    def load(self) -> bool:
        """
        Loads the YOLOv8 model weights.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"YOLOv8 model loaded from '{YOLO_MODEL_PATH}'")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Runs person detection on a single frame.

        Parameters:
            frame: Preprocessed BGR numpy array.

        Returns:
            numpy array of shape (N, 5) where each row is
            [x1, y1, x2, y2, confidence] for one detected person.
            Returns empty array if no persons found.
        """
        if self.model is None:
            logger.error("Model not loaded. Call load() first.")
            return np.empty((0, 5))

        try:
            results = self.model(
                frame,
                conf=YOLO_CONF_THRESHOLD,
                iou=YOLO_IOU_THRESHOLD,
                classes=[YOLO_PERSON_CLASS_ID],
                verbose=False,
            )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return np.empty((0, 5))

            # Extract xyxy + confidence as a numpy array
            xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
            conf = boxes.conf.cpu().numpy()        # (N,)
            detections = np.hstack([xyxy, conf[:, None]])  # (N, 5)

            logger.debug(f"Detected {len(detections)} person(s)")
            return detections

        except Exception as e:
            logger.error(f"Detection error on frame: {e}")
            return np.empty((0, 5))
