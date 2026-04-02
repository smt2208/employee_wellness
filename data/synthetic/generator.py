# data/synthetic/generator.py
# Generates synthetic frame sequences that mimic what the pipeline backend
# receives from real CCTV footage.
#
# Each synthetic frame is a list of person dicts — exactly the format
# that PostureClassifier.classify() expects as input.
# Keypoints are geometrically correct so the classifier produces the
# intended posture labels, making pipeline verification meaningful.
#
# WHAT THIS REPLACES:
#   Real pipeline: VideoReader → YOLO → BoTSORT → RTMPose → PostureClassifier
#   Synthetic:     Generator   → (skip detection/tracking/pose) → PostureClassifier
#
# Usage:
#   python -m data.synthetic.generator
#   → Default verification run (more data, still faster than full)
#   python -m data.synthetic.generator --mode quick
#   → Small smoke test
#   python -m data.synthetic.generator --mode full
#   → Full-length session
#   → Saves JSON files to data/synthetic/output/
#   → Runs the pipeline on them and prints index results

import json
import os
import random
import sys
import time
import argparse
from datetime import datetime, timedelta
from typing import Optional

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from data.synthetic.profiles     import PROFILES
from data.synthetic.keypoint_builder import build_keypoints
from config.settings import (
    TARGET_FPS,
    SHOULDER_SLOPE_TENSE_THRESHOLD,
    FATIGUE_LABELS,
    STRESS_LABELS,
    ENGAGEMENT_LABELS,
)
from utils.logger    import get_logger

logger = get_logger(__name__)

# Output directory for generated JSON files
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

# Fast smoke-test defaults (used by CLI quick mode)
QUICK_DURATION_HOURS = 0.03   # ~108 seconds at TARGET_FPS
QUICK_NUM_PERSONS    = 2
QUICK_MAX_FRAMES     = 300

# Verification defaults (enough temporal data for more stable classifier output)
VERIFY_DURATION_HOURS = 0.55   # ~33 minutes at TARGET_FPS
VERIFY_NUM_PERSONS    = 2
VERIFY_MAX_FRAMES     = 30000


def _label_names(label_table: list) -> list:
    """Returns ordered label names from a (threshold, label) table."""
    return [label for _, label in label_table]


def _label_match_type(expected_label: str, computed_label: str, label_table: list) -> str:
    """
    Returns match type for two labels on the same ordered severity scale.

    Values:
      - "exact"    : same label
      - "adjacent" : one bucket away on the label scale
      - "mismatch" : farther than one bucket
      - "invalid"  : unknown label not present in the table
    """
    labels = _label_names(label_table)
    if expected_label not in labels or computed_label not in labels:
        return "invalid"
    if expected_label == computed_label:
        return "exact"
    if abs(labels.index(expected_label) - labels.index(computed_label)) <= 1:
        return "adjacent"
    return "mismatch"


def _validate_profile_expected_labels():
    """Ensures synthetic profile expected labels exist in current settings tables."""
    label_tables = {
        "fatigue": FATIGUE_LABELS,
        "stress": STRESS_LABELS,
        "engagement": ENGAGEMENT_LABELS,
    }

    errors = []
    for profile_name, profile in PROFILES.items():
        expected = profile.get("expected", {})
        for metric_name, label_table in label_tables.items():
            expected_label = expected.get(metric_name)
            valid_labels = _label_names(label_table)
            if expected_label not in valid_labels:
                errors.append(
                    f"profile='{profile_name}' metric='{metric_name}' expected='{expected_label}' valid={valid_labels}"
                )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            "Synthetic profile expected labels are out of sync with current settings:\n"
            f"  - {joined}"
        )


def _sample_label(prob_dict: dict) -> str:
    """
    Samples a label from a probability distribution dict.

    Parameters:
        prob_dict: {label: probability} — probabilities should sum to ~1.0

    Returns:
        Sampled label string.
    """
    labels = list(prob_dict.keys())
    probs  = list(prob_dict.values())
    # Normalize in case probabilities don't sum to exactly 1.0
    total  = sum(probs)
    probs  = [p / total for p in probs]
    return random.choices(labels, weights=probs, k=1)[0]


def _sample_bool(rate: float) -> bool:
    """Returns True with probability equal to rate."""
    return random.random() < rate


def _sample_temporal_label(
    prob_dict: dict,
    previous_label: Optional[str],
    switch_rate: float,
) -> str:
    """
    Samples labels with temporal persistence.

    Lower switch_rate produces steadier behaviour, while higher values
    produce rapid shifts (useful for stressed/agitated profiles).
    """
    rate = max(0.0, min(1.0, switch_rate))
    if previous_label is None or random.random() < rate:
        return _sample_label(prob_dict)
    return previous_label


def _make_bbox(centre_x: float, centre_y: float,
               width: float = 110, height: float = 330) -> list:
    """
    Constructs a bounding box around a centre point.

    Parameters:
        centre_x: Horizontal centre in pixels.
        centre_y: Vertical centre in pixels.
        width:    Box width in pixels.
        height:   Box height in pixels.

    Returns:
        [x1, y1, x2, y2]
    """
    return [
        centre_x - width  / 2,
        centre_y - height / 2,
        centre_x + width  / 2,
        centre_y + height / 2,
    ]


def generate_person_frames(
    track_id:       int,
    profile_name:   str,
    num_frames:     int,
    start_x:        float = 640.0,
    start_y:        float = 360.0,
) -> list:
    """
    Generates a sequence of person dicts for one employee over num_frames.

    Each dict is the format PostureClassifier.classify() receives — it has
    track_id, bbox, and keypoints. The classifier adds posture/activity/signals.

    Parameters:
        track_id:     Persistent ID for this simulated employee.
        profile_name: Key from PROFILES dict.
        num_frames:   Number of frames to generate.
        start_x:      Starting horizontal position of person in frame.
        start_y:      Starting vertical position of person in frame.

    Returns:
        List of frame dicts, each containing one person.
    """
    profile = PROFILES[profile_name]
    frames  = []

    cx, cy = start_x, start_y   # current bbox centre
    prev_posture: Optional[str] = None
    prev_activity: Optional[str] = None

    posture_switch_rate = profile.get("posture_shift_rate", 0.1)
    # Activity tends to change slower than frame-by-frame random sampling.
    activity_switch_rate = max(0.08, profile.get("restless_rate", 0.1) * 0.6)

    for i in range(num_frames):
        # ── Sample posture and activity with temporal continuity ─────────────
        posture = _sample_temporal_label(
            profile["posture_probs"], prev_posture, posture_switch_rate
        )
        activity = _sample_temporal_label(
            profile["activity_probs"], prev_activity, activity_switch_rate
        )
        prev_posture = posture
        prev_activity = activity

        # ── Sample binary wellness signals from profile rates ─────────────────
        shoulder_tension = _sample_bool(profile["shoulder_tension_rate"])
        restless         = _sample_bool(profile["restless_rate"])
        posture_shift    = _sample_bool(profile["posture_shift_rate"])
        frozen           = _sample_bool(profile["frozen_rate"])
        head_level       = _sample_bool(profile["head_level_rate"])

        # Enforce logical consistency: frozen and restless can't both be True
        if frozen and restless:
            frozen = False

        # For head_down posture, head_level must be False
        if posture == "head_down":
            head_level = False

        # ── Compute realistic bbox movement ───────────────────────────────────
        mean_disp, std_disp = profile["displacement"]
        displacement = max(0.0, random.gauss(mean_disp, std_disp))

        # Inject profile behaviour into motion so classifier sees intended dynamics.
        if frozen:
            displacement *= random.uniform(0.02, 0.20)
        elif restless:
            displacement += abs(random.gauss(mean_disp * 0.6, std_disp + 3.0))

        # Direction: mostly staying in place, with occasional random drift
        cx += displacement * 0.3 * (1 if random.random() > 0.5 else -1)
        cy += displacement * 0.1 * (1 if random.random() > 0.5 else -1)

        # Keep person within frame bounds (100px margins)
        cx = max(200, min(1080, cx))
        cy = max(200, min(520,  cy))

        bbox = _make_bbox(cx, cy)

        # ── Build keypoints geometrically matching the sampled posture ────────
        keypoints = build_keypoints(posture)

        # Offset keypoints to match bbox position (builder uses fixed centre 640,360)
        kp_offset_x = cx - 640
        kp_offset_y = cy - 360
        for kp in keypoints.values():
            kp["x"] += kp_offset_x
            kp["y"] += kp_offset_y

        # Force detectable shoulder asymmetry when tension is sampled.
        if (
            shoulder_tension
            and "left_shoulder" in keypoints
            and "right_shoulder" in keypoints
        ):
            min_gap = SHOULDER_SLOPE_TENSE_THRESHOLD + 2
            gap = random.uniform(min_gap, min_gap + 10)
            keypoints["left_shoulder"]["y"] -= gap / 2
            keypoints["right_shoulder"]["y"] += gap / 2

        frames.append({
            "frame_index":      i,
            "track_id":         track_id,
            "bbox":             bbox,
            "confidence":       round(random.uniform(0.75, 0.97), 3),
            "keypoints":        keypoints,
            # Pre-sampled signals (used to verify pipeline output matches profile)
            "_synthetic_posture":         posture,
            "_synthetic_activity":        activity,
            "_synthetic_shoulder_tension":shoulder_tension,
            "_synthetic_restless":        restless,
            "_synthetic_posture_shift":   posture_shift,
            "_synthetic_frozen":          frozen,
            "_synthetic_head_level":      head_level,
        })

    logger.info(
        f"Generated {num_frames} frames for track_id={track_id} "
        f"profile='{profile_name}'"
    )
    return frames


def generate_session(
    duration_hours: float = 2.0,
    num_persons:    int   = 4,
    max_frames: Optional[int] = None,
) -> dict:
    """
    Generates a full office session with multiple employees,
    each assigned a different wellness profile.

    Parameters:
        duration_hours: Length of the simulated session in hours.
        num_persons:    Number of employees to simulate.
        max_frames:     Optional cap on generated frames per person.

    Returns:
        Session dict containing:
            - metadata
            - per-person frame sequences
            - expected wellness outcomes for verification
    """
    total_frames = int(duration_hours * 3600 * TARGET_FPS)
    total_frames = max(1, total_frames)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    profile_names = list(PROFILES.keys())

    # Assign one profile per person (cycle if more persons than profiles)
    assignments = {}
    for i in range(num_persons):
        pid          = i + 1
        profile_name = profile_names[i % len(profile_names)]
        assignments[pid] = profile_name

    logger.info(
        f"Generating session | duration={duration_hours}h | "
        f"frames={total_frames} | persons={num_persons}"
    )

    session = {
        "metadata": {
            "generated_at":            datetime.now().isoformat(),
            "duration_hours":          duration_hours,
            "effective_duration_hours":round(total_frames / (3600 * TARGET_FPS), 4),
            "total_frames":            total_frames,
            "fps":                     TARGET_FPS,
            "num_persons":             num_persons,
        },
        "persons": [],
    }

    # Space persons across the frame horizontally
    x_positions = [200 + i * (880 // max(num_persons - 1, 1))
                   for i in range(num_persons)]

    for i, (pid, profile_name) in enumerate(assignments.items()):
        profile = PROFILES[profile_name]
        frames  = generate_person_frames(
            track_id=pid,
            profile_name=profile_name,
            num_frames=total_frames,
            start_x=x_positions[i],
            start_y=360.0,
        )
        session["persons"].append({
            "track_id":    pid,
            "profile":     profile_name,
            "description": profile["description"],
            "expected":    profile["expected"],
            "frames":      frames,
        })

    return session


def save_session(session: dict, filename: Optional[str] = None) -> str:
    """
    Saves a generated session to a JSON file.

    Parameters:
        session:  Session dict from generate_session().
        filename: Output filename. Auto-generated from timestamp if None.

    Returns:
        Full path to the saved file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_session_{ts}.json"

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(session, f, indent=2)

    logger.info(f"Session saved to: {filepath}")
    return filepath


def run_pipeline_on_session(session: dict) -> list:
    """
    Feeds the synthetic frames through the PostureClassifier and
    TemporalFusion stages — exactly as the real pipeline would.

    Bypasses VideoReader, YOLO, BoTSORT, and RTMPose (not needed for testing
    the classifier and fusion logic — those are the stages being validated).

    Parameters:
        session: Session dict from generate_session().

    Returns:
        List of per-person result dicts with computed wellness indices
        and comparison against expected outcomes.
    """
    _validate_profile_expected_labels()

    from src.classifier.posture_classifier import PostureClassifier
    from src.fusion.temporal_fusion        import TemporalFusion

    classifier = PostureClassifier()
    fusion     = TemporalFusion()

    total_frames = session["metadata"]["total_frames"]
    num_persons  = session["metadata"]["num_persons"]

    logger.info(
        f"Running pipeline on {num_persons} persons × {total_frames} frames..."
    )

    # Feed all frames to classifier and fusion
    for frame_idx in range(total_frames):
        frame_persons = []

        for person_data in session["persons"]:
            if frame_idx >= len(person_data["frames"]):
                continue
            raw_frame = person_data["frames"][frame_idx]

            # Strip synthetic metadata before passing to classifier
            person_input = {
                "track_id":  raw_frame["track_id"],
                "bbox":      raw_frame["bbox"],
                "confidence":raw_frame["confidence"],
                "keypoints": raw_frame["keypoints"],
            }
            classified = classifier.classify(person_input)
            frame_persons.append(classified)

        if frame_persons:
            active_ids = {p["track_id"] for p in frame_persons}
            fusion.update(frame_persons)
            fusion.remove_lost_tracks(active_ids)

        # Progress log every 10% of frames
        if total_frames > 100 and frame_idx % (total_frames // 10) == 0:
            pct = int(100 * frame_idx / total_frames)
            logger.info(f"  Processing... {pct}%")

    # Collect results and compare against expected
    all_indices = fusion.get_all_indices()
    results     = []

    for person_data in session["persons"]:
        pid      = person_data["track_id"]
        expected = person_data["expected"]

        indices = next(
            (idx for idx in all_indices if idx["track_id"] == pid), None
        )

        if indices is None:
            results.append({
                "track_id": pid,
                "profile":  person_data["profile"],
                "error":    "No indices computed",
            })
            continue

        # Use the "full" rolling window (most data, most stable)
        full_window = indices["windows"].get("full", {})

        computed = {
            "fatigue_label":    full_window.get("fatigue_label",    "N/A"),
            "stress_label":     full_window.get("stress_label",     "N/A"),
            "engagement_label": full_window.get("engagement_label", "N/A"),
        }

        # Compare computed labels against expected.
        match_types = {
            "fatigue": _label_match_type(
                expected["fatigue"], computed["fatigue_label"], FATIGUE_LABELS
            ),
            "stress": _label_match_type(
                expected["stress"], computed["stress_label"], STRESS_LABELS
            ),
            "engagement": _label_match_type(
                expected["engagement"], computed["engagement_label"], ENGAGEMENT_LABELS
            ),
        }

        strict_matches = {
            metric: match_type == "exact"
            for metric, match_type in match_types.items()
        }
        tolerant_matches = {
            metric: match_type in ("exact", "adjacent")
            for metric, match_type in match_types.items()
        }
        strict_all_match = all(strict_matches.values())
        all_match = all(tolerant_matches.values())

        results.append({
            "track_id":    pid,
            "profile":     person_data["profile"],
            "description": person_data["description"],
            "expected":    expected,
            "computed":    computed,
            "match_types": match_types,
            "strict_matches": strict_matches,
            "matches":     tolerant_matches,
            "strict_all_match": strict_all_match,
            "all_match":   all_match,
        })

    return results


def print_results(results: list):
    """
    Prints a formatted verification report to the terminal.

    Parameters:
        results: List of result dicts from run_pipeline_on_session().
    """
    print("\n" + "=" * 70)
    print("  SYNTHETIC DATA PIPELINE VERIFICATION REPORT")
    print("=" * 70)

    passed = 0
    failed = 0
    exact_passed = 0

    for r in results:
        if r.get("all_match"):
            if r.get("strict_all_match"):
                status = "PASS ✓ (exact)"
                exact_passed += 1
            else:
                status = "PASS ~ (adjacent)"
        else:
            status = "FAIL ✗"
        if r.get("all_match"):
            passed += 1
        else:
            failed += 1

        print(f"\nPerson {r['track_id']}  |  Profile: {r['profile']}")
        print(f"  {r.get('description', '')}")
        print(f"  Status: {status}")

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        exp = r["expected"]
        cmp = r["computed"]

        print(f"  {'Index':<15} {'Expected':<25} {'Computed':<25}  Match")
        print(f"  {'-'*15} {'-'*25} {'-'*25}  -----")

        for idx_name in ("fatigue", "stress", "engagement"):
            e_label = exp.get(idx_name, "?")
            c_label = cmp.get(f"{idx_name}_label", "?")
            match_type = r.get("match_types", {}).get(idx_name, "mismatch")
            if match_type == "exact":
                match = "Exact"
            elif match_type == "adjacent":
                match = "Adjacent"
            else:
                match = "Mismatch"

            print(
                f"  {idx_name.capitalize():<15} {e_label:<25} {c_label:<25}  {match}"
            )

    print("\n" + "=" * 70)
    print(f"  RESULT:  {passed} passed  |  {failed} failed  |  "
          f"{len(results)} total persons")
    print(f"  Exact matches: {exact_passed}/{len(results)} persons")
    print("=" * 70 + "\n")


def _build_session_from_mode(
    mode: str,
    duration_hours: Optional[float],
    num_persons: Optional[int],
    max_frames: Optional[int],
) -> dict:
    """
    Builds a quick smoke-test, medium verification, or full session.

    verify mode balances runtime and reliability by providing enough temporal
    samples for classifier and fusion logic to stabilise.
    """
    if mode == "quick":
        duration = QUICK_DURATION_HOURS if duration_hours is None else duration_hours
        persons = QUICK_NUM_PERSONS if num_persons is None else num_persons
        frame_cap = QUICK_MAX_FRAMES if max_frames is None else max_frames
    elif mode == "verify":
        duration = VERIFY_DURATION_HOURS if duration_hours is None else duration_hours
        persons = VERIFY_NUM_PERSONS if num_persons is None else num_persons
        frame_cap = VERIFY_MAX_FRAMES if max_frames is None else max_frames
    else:
        duration = 2.0 if duration_hours is None else duration_hours
        persons = 4 if num_persons is None else num_persons
        frame_cap = max_frames

    return generate_session(
        duration_hours=duration,
        num_persons=persons,
        max_frames=frame_cap,
    )


def _parse_args() -> argparse.Namespace:
    """Parses CLI arguments for quick/verify/full synthetic test runs."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic sessions for wellness pipeline verification."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "verify", "full"],
        default="verify",
        help="quick: fast smoke test, verify: more data for stable checks (default), full: long session.",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Override session duration in hours.",
    )
    parser.add_argument(
        "--num-persons",
        type=int,
        default=None,
        help="Override number of persons to simulate.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap generated frames per person for faster test cycles.",
    )

    args = parser.parse_args()

    if args.duration_hours is not None and args.duration_hours <= 0:
        parser.error("--duration-hours must be > 0")
    if args.num_persons is not None and args.num_persons <= 0:
        parser.error("--num-persons must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")

    return args


if __name__ == "__main__":
    print("\nWellness Pipeline — Synthetic Data Generator")
    print("─" * 50)

    args = _parse_args()
    _validate_profile_expected_labels()

    # Verify mode is the default: still much faster than full, but with
    # enough temporal data for classifier checks to be meaningful.
    session = _build_session_from_mode(
        mode=args.mode,
        duration_hours=args.duration_hours,
        num_persons=args.num_persons,
        max_frames=args.max_frames,
    )
    meta = session["metadata"]
    print(
        f"Mode={args.mode} | Persons={meta['num_persons']} | "
        f"Frames/person={meta['total_frames']} | "
        f"Effective duration={meta['effective_duration_hours']}h"
    )

    filepath = save_session(session)
    print(f"Session data saved to: {filepath}")

    # Run the pipeline on the generated data
    results = run_pipeline_on_session(session)

    # Save results alongside session data
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(OUTPUT_DIR, f"verification_{ts}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print_results(results)
    print(f"Full results saved to: {results_path}")
