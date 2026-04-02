# src/preprocessing/video_reader.py
# Handles video capture from file or webcam and applies
# basic preprocessing to each frame before detection.

import cv2
import numpy as np
from config.settings import TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoReader:
    """
    Opens a video source and yields preprocessed frames one at a time.

    Supports:
        - Video files (.mp4, .avi, etc.)
        - Webcam (pass source=0)
        - Skips frames to match TARGET_FPS and reduce compute load
    """

    def __init__(self, source):
        """
        Parameters:
            source: File path string or integer camera index (0 for default webcam).
        """
        self.source = source
        self.cap    = None
        self._frame_skip = 1   # Calculated once the stream is opened

    def open(self) -> bool:
        """
        Opens the video source.

        Returns:
            True if opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        # Calculate how many source frames to skip to hit TARGET_FPS
        self._frame_skip = max(1, int(round(source_fps / TARGET_FPS)))

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Opened source '{self.source}' | "
            f"Source FPS: {source_fps:.1f} | "
            f"Processing every {self._frame_skip} frame(s) | "
            f"Total frames: {total_frames}"
        )
        return True

    def read_frames(self):
        """
        Generator that yields (frame_index, preprocessed_frame) tuples.
        Skips frames automatically to match TARGET_FPS.

        Yields:
            Tuple of (int frame_index, np.ndarray preprocessed BGR frame)
        """
        if self.cap is None:
            logger.error("VideoReader not opened. Call open() first.")
            return

        frame_index = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                logger.info("End of video stream reached.")
                break

            # Only process every Nth frame
            if frame_index % self._frame_skip == 0:
                processed = self._preprocess(frame)
                yield frame_index, processed

            frame_index += 1

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies basic preprocessing to a raw frame.

        Steps:
            1. Resize to standard resolution
            2. Denoise with a fast Gaussian blur
            3. Normalize brightness with CLAHE

        Parameters:
            frame: Raw BGR frame from OpenCV.

        Returns:
            Preprocessed BGR frame as numpy array.
        """
        # Step 1: Resize
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Step 2: Gentle denoise (kernel 3x3 is fast and non-destructive)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        # Step 3: Brightness normalization using CLAHE on the L channel only.
        # clipLimit=1.5 (reduced from 2.0) — gentler correction that fixes
        # CCTV exposure unevenness while preserving colour fidelity for
        # BoTSORT's Re-ID model. Higher values shift colours enough that
        # the same person's appearance embedding may not match on re-entry.
        try:
            lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l       = clahe.apply(l)
            lab     = cv2.merge([l, a, b])
            frame   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"CLAHE preprocessing failed, using fallback frame: {e}")

        return frame

    def release(self):
        """Releases the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source released.")
