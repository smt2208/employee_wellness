# main.py
# Entry point for the wellness pipeline.
# Run this file to start processing a video source.
#
# Usage:
#   python main.py                        → uses default from config (webcam or file)
#   python main.py --source video.mp4     → processes a video file
#   python main.py --source 0             → uses webcam 0

import argparse
import sys

from src.pipeline    import WellnessPipeline
from config.settings import VIDEO_SOURCE
from utils.logger    import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Driven Employee Wellness Monitoring Pipeline"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index (default: from config/settings.py)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    source = args.source if args.source is not None else VIDEO_SOURCE

    # If source is a digit string (e.g. "0"), convert to int for OpenCV
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    logger.info(f"Starting Wellness Pipeline | source='{source}'")

    pipeline = WellnessPipeline(video_source=source)

    if not pipeline.initialise():
        logger.error("Pipeline initialisation failed. Exiting.")
        sys.exit(1)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as e:
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
