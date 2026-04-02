# SYSTEM CONTEXT PROMPT — AI-Driven Employee Wellness Monitoring Pipeline
# Copy this entire file's contents into GitHub Copilot Agent's system/context prompt.
# This gives Copilot complete awareness of the project for all future development.

---

## PROJECT IDENTITY

You are assisting with a research-led Proof of Concept (PoC) called the
**AI-Driven Employee Wellness Monitoring Pipeline**, built for a utility sector
organisation. The system uses computer vision and behavioural analytics
(kinesics) to passively monitor employee mental wellness — specifically fatigue,
stress/agitation, and engagement — using CCTV camera footage. It is explicitly
NOT a performance monitoring or surveillance system. It is a well-being support
tool that generates non-prescriptive, curated insights for managers.

**Intern:** Shyam (computer vision intern at a data science lab)
**Mentor:** Assigned supervisor at the data science lab
**Client:** Utility sector organisation (prospective)
**Local project path:** `C:\Users\User\Downloads\wellness_pipeline`

---

## PROBLEM STATEMENT SUMMARY

Organisations rely on self-reported surveys and periodic feedback to gauge
employee sentiment. These are episodic, subjective, and miss real-time
emotional changes. This system proposes using AI-based analysis of non-verbal
behavioural cues — posture and activity — to provide continuous, individual-level
wellness insights that are more reliable and timely than traditional methods.

The system is scoped as a PoC over approximately two weeks of observational data.
Data sources are existing CCTV infrastructure and/or an opt-in mobile application.

---

## CRITICAL DESIGN DECISIONS (do not reverse without strong reason)

1. **Facial emotion recognition is explicitly removed.**
   CCTV cameras have insufficient resolution, unfavourable angles, motion blur,
   and occlusions for reliable facial emotion detection. The system operates
   entirely on body posture and activity signals. Do not suggest re-adding face
   detection unless CCTV quality is confirmed adequate.

2. **No fine-tuning of models.**
   The pipeline uses only pretrained models to keep the PoC clean and achievable
   within an internship timeline. Fine-tuning on CCTV posture data is documented
   as future work.

3. **BoTSORT over ByteTrack for tracking.**
   Office employees frequently leave the camera frame for extended periods
   (meetings, breaks, etc.). ByteTrack is purely motion-based and loses tracks
   after a buffer period, wiping wellness history. BoTSORT adds appearance-based
   Re-ID (OSNet embeddings) that allows re-identification of the same person
   when they re-enter the frame, restoring their track ID and history.

4. **Posture-only wellness signals.**
   All three mental wellness indices are computed exclusively from skeletal
   keypoints and movement patterns. No audio, no facial, no physiological data.

5. **CLAHE clipLimit = 1.5 (not 2.0).**
   Reduced from 2.0 to preserve colour fidelity for BoTSORT's Re-ID model.
   Higher values shift colour channels enough to degrade appearance matching
   when a person re-enters the frame.

6. **Three indices, not a single wellness score.**
   A single score is too vague for clinical or managerial use. Three separate
   indices (Fatigue, Stress/Agitation, Engagement) with clear governing signals
   are more interpretable, explainable, and actionable.

7. **Tumbling + rolling window hybrid, not rolling-only.**
   A rolling-only approach blends all history together, making morning and
   afternoon patterns indistinguishable. A 2-hour tumbling window creates clean
   sealed session blocks aligned to real office work patterns. Rolling sub-windows
   operate within each block for multi-resolution trend analysis.

---

## FINAL TECHNICAL ARCHITECTURE

### Pipeline Stages (in order)

```
Stage 1: OpenCV Preprocessing
  → Frame extraction from video source
  → Resize to 1280×720
  → Gaussian blur (3×3 kernel) — sensor noise removal
  → CLAHE (clipLimit=1.5, tileGridSize=8×8) on L channel — brightness normalisation
  → Output: clean BGR frame (numpy array)

Stage 2: YOLOv8 Person Detection
  → Model: yolov8n.pt (pretrained, COCO)
  → Detects only class 0 (person)
  → Confidence threshold: 0.5, IOU threshold: 0.4
  → Output: numpy array shape (N, 5) — [x1, y1, x2, y2, confidence]

Stage 3: BoTSORT Tracking
  → Library: boxmot
  → Re-ID model: osnet_x0_25_msmt17.pt (auto-downloaded by boxmot)
  → Assigns persistent track_id per person across all frames
  → Track buffer: 150 frames (10 seconds at 15fps)
  → Re-ID enables re-identification after long frame absences (meetings, breaks)
  → Output: list of {track_id, bbox, confidence} dicts

Stage 4: RTMPose Pose Estimation
  → Library: rtmlib
  → Model: RTMPose body (17-point COCO keypoints)
  → Top-down mode: runs inside each person's bounding box
  → Keypoint confidence threshold: 0.4 (discard low-confidence points)
  → Output: 17 named keypoints per person {name: {x, y, conf}}
  → Chosen over YOLOv8-Pose (single pass but less accurate on CCTV angles)
     and MediaPipe (single-person only)

Stage 5: PostureClassifier (rule-based)
  → Computes spine angle: atan2(|dx|, |dy|) of mid_hip→mid_shoulder vector
  → Computes head tilt: atan2(|dx|, |dy|) of mid_shoulder→nose vector
  → Computes shoulder slope: |left_shoulder.y - right_shoulder.y|
  → Computes bbox displacement for activity classification
  → Computes movement variance (std dev) for restlessness/frozen signals
  → Computes posture shift count for agitation signal
  → Output: posture label + activity label + 5 binary wellness signals per person

Stage 6: TemporalFusion
  → Converts per-frame signals into three mental wellness indices
  → 2-hour tumbling window with 4 rolling sub-windows inside
  → Output: Fatigue Index, Stress Index, Engagement Index (0-100 each)

Stage 7: Output
  → Annotated video saved to output/results/
  → JSON session log saved to output/results/
  → Console summary of final indices per person
```

---

## POSTURE & ACTIVITY LABELS

### Posture Labels (from PostureClassifier)
| Label | Condition | Primary Index |
|---|---|---|
| `upright` | spine angle ≤ 15° | Engagement ↑ |
| `mild_slouch` | 15° < spine angle ≤ 30° | Fatigue mild |
| `slouching` | spine angle > 30° | Fatigue ↑ |
| `head_down` | head tilt > 20° (overrides spine) | Fatigue ↑↑ |
| `unknown` | required keypoints missing | Neutral |

### Activity Labels (from PostureClassifier)
| Label | Condition | Primary Index |
|---|---|---|
| `walking` | displacement > 35px/frame | Engagement ↑ |
| `standing` | displacement < 8px, not sitting | Engagement neutral |
| `sitting_active` | displacement 8-35px, sitting | Engagement ↑ |
| `sitting_idle` | displacement < 8px, sitting | Fatigue mild |
| `prolonged_idle` | idle > 30 continuous minutes | Fatigue ↑↑ |
| `unknown` | cannot determine | Neutral |

### Binary Wellness Signals (from PostureClassifier)
| Signal | How computed | Primary Index |
|---|---|---|
| `shoulder_tension` | shoulder height asymmetry ≥ 15px | Stress ↑ |
| `restless` | displacement std-dev ≥ 12.0 over 60 frames | Stress ↑ |
| `posture_shift` | ≥ 4 unique posture labels in 90 frames | Stress ↑ |
| `frozen` | displacement std-dev ≤ 2.0 over 60 frames | Fatigue ↑ |
| `head_level` | head tilt ≤ 20° (True = good) | Engagement ↑ |

---

## THREE MENTAL WELLNESS INDICES

### Fatigue Index (0-100, higher = more fatigued)
```
Components:
  prolonged_idle_rate   × 0.35   (% frames where activity == prolonged_idle)
  slouching_rate        × 0.25   (% frames where posture in [slouching, head_down])
  low_activity_rate     × 0.20   (% frames where activity in [sitting_idle, prolonged_idle])
  posture_rigidity_rate × 0.20   (% frames where frozen == True)

Labels:  0-24% → Minimal Fatigue
        25-49% → Low Fatigue
        50-74% → Moderate Fatigue
        75-100%→ High Fatigue
```

### Stress / Agitation Index (0-100, higher = more stressed)
```
Components:
  restlessness_rate     × 0.35   (% frames where restless == True)
  shoulder_tension_rate × 0.25   (% frames where shoulder_tension == True)
  posture_shift_rate    × 0.25   (% frames where posture_shift == True)
  rigid_upright_rate    × 0.15   (% frames where upright AND frozen both True)

Labels:  0-24% → Minimal Stress
        25-49% → Low Stress
        50-74% → Moderate Stress
        75-100%→ High Stress
```

### Engagement Index (0-100, higher = more engaged)
```
Components:
  upright_rate    × 0.35   (% frames where posture == upright)
  active_rate     × 0.35   (% frames where activity in [walking, standing, sitting_active])
  head_level_rate × 0.30   (% frames where head_level == True)

Labels:  0-24% → Disengaged
        25-49% → Low Engagement
        50-74% → Moderately Engaged
        75-100%→ Highly Engaged
```

---

## TEMPORAL WINDOW ARCHITECTURE

### Structure
```
TUMBLING WINDOW: 2 hours (7200 seconds)
  → Hard reset every 2 hours
  → At reset: final indices computed and sealed into completed_blocks[]
  → Buffer cleared, new window starts fresh
  → Aligns to real office blocks: 08-10, 10-12, 12-14, 14-16, 16-18

ROLLING SUB-WINDOWS (computed on-demand within current tumbling block):
  micro  → last 10 minutes  (immediate state, catches quick mood changes)
  short  → last 30 minutes  (session-level trend)
  medium → last 1 hour      (half-block trend)
  full   → last 2 hours     (entire current tumbling block)
```

### How Computation Works
Every frame, one `Observation` object is appended to a deque buffer.
Each Observation stores a timestamp and all binary signal values for that frame.
When `get_indices()` is called, it:
1. Filters the buffer to observations within each rolling window's time range
2. Counts how many observations have each signal = True
3. Divides by total observations to get rates
4. Applies index weights to compute 0-100 scores
No timers. No caching. All windows computed fresh on every call.

### Output Structure per Person per Frame
```python
{
  "track_id": int,
  "windows": {
    "micro":  {"fatigue": float, "stress": float, "engagement": float,
               "fatigue_label": str, "stress_label": str,
               "engagement_label": str, "sample_count": int},
    "short":  {...},
    "medium": {...},
    "full":   {...},
  },
  "completed_blocks": int,
}
```

---

## PROJECT FILE STRUCTURE

```
wellness_pipeline/
│
├── main.py                              ← Entry point. Run this.
│
├── requirements.txt                     ← All dependencies
│
├── config/
│   └── settings.py                      ← ALL thresholds, weights, paths.
│                                          Edit here to tune behaviour.
│
├── src/
│   ├── pipeline.py                      ← Orchestrator. Connects all stages.
│   │                                      Only file that imports from all stages.
│   │
│   ├── preprocessing/
│   │   └── video_reader.py              ← Stage 1: OpenCV frame extraction
│   │
│   ├── detection/
│   │   └── person_detector.py           ← Stage 2: YOLOv8 person detection
│   │
│   ├── tracking/
│   │   └── tracker.py                   ← Stage 3: BoTSORT with Re-ID
│   │
│   ├── pose/
│   │   └── pose_estimator.py            ← Stage 4: RTMPose keypoint estimation
│   │
│   ├── classifier/
│   │   └── posture_classifier.py        ← Stage 5: Rule-based signals
│   │
│   └── fusion/
│       └── temporal_fusion.py           ← Stage 6: Tumbling+rolling wellness indices
│
├── data/
│   └── synthetic/
│       ├── profiles.py                  ← 4 employee behaviour scenarios
│       ├── keypoint_builder.py          ← Geometrically correct skeleton generator
│       ├── generator.py                 ← Generates sessions + runs verification
│       └── output/                      ← Generated JSON files saved here
│
├── output/
│   └── results/                         ← Annotated videos + session JSON logs
│
└── utils/
    └── logger.py                        ← Shared logger (use get_logger(__name__))
```

### Separation of Concerns Rule
Individual stage modules (detection, tracking, pose, classifier, fusion) NEVER
import from each other. Only `src/pipeline.py` imports from all stages.
This prevents circular imports and keeps each stage independently testable.

---

## SYNTHETIC DATA SYSTEM

### Purpose
Tests the PostureClassifier and TemporalFusion stages without real CCTV footage.
Bypasses VideoReader, YOLO, BoTSORT, and RTMPose entirely.
Feeds geometrically correct keypoints directly into the classifier.

### 4 Profiles (in data/synthetic/profiles.py)
| Profile | Expected Fatigue | Expected Stress | Expected Engagement |
|---|---|---|---|
| `healthy_engaged` | Minimal Fatigue | Minimal Stress | Highly Engaged |
| `fatigued` | High Fatigue | Low Stress | Low Engagement |
| `stressed` | Low Fatigue | High Stress | Moderately Engaged |
| `disengaged` | Moderate Fatigue | Low Stress | Disengaged |

### Keypoint Validity Guarantee
Keypoints are not randomly generated. `keypoint_builder.py` reverse-engineers
the exact geometric formula used by PostureClassifier to ensure that generated
keypoints produce the intended posture label when the classifier processes them.

For spine angle:
  dx = |dy| × tan(target_angle_degrees)
  Shift entire upper body by dx to produce exact target spine angle.

### Run Verification
```bash
cd C:\Users\User\Downloads\wellness_pipeline
python -m data.synthetic.generator
```
Outputs a PASS/FAIL report comparing pipeline-computed index labels
against profile-expected labels for all 4 employee profiles.

---

## DEPENDENCIES

```
ultralytics>=8.0.0     # YOLOv8
boxmot>=10.0.0         # BoTSORT (includes Re-ID models)
rtmlib>=0.0.13         # RTMPose
opencv-python>=4.8.0   # Video + preprocessing
numpy>=1.24.0          # Arrays
onnxruntime>=1.16.0    # RTMPose inference backend
gdown>=4.7.0           # Re-ID model download helper
```

Install: `pip install -r requirements.txt`
Run:     `python main.py --source video.mp4`  or  `python main.py --source 0`

---

## KEY CONFIGURATION VALUES (config/settings.py)

| Parameter | Value | Why |
|---|---|---|
| TARGET_FPS | 15 | Process every 2nd frame at 30fps source — saves compute |
| FRAME_WIDTH/HEIGHT | 1280×720 | Consistent model input size |
| YOLO_CONF_THRESHOLD | 0.5 | Min confidence to accept a person detection |
| TRACK_BUFFER | 150 | 10 seconds at 15fps before losing a track |
| REID_MODEL_PATH | osnet_x0_25_msmt17.pt | Appearance Re-ID weights |
| KEYPOINT_CONF_THRESHOLD | 0.4 | Discard unreliable keypoints |
| SPINE_UPRIGHT_MAX | 15° | Max spine angle for "upright" |
| SPINE_MILD_SLOUCH_MAX | 30° | Max spine angle for "mild_slouch" |
| HEAD_LEVEL_MAX | 20° | Max head tilt for "head_level" |
| SHOULDER_SLOPE_TENSE_THRESHOLD | 15px | Min asymmetry for tension |
| MOVEMENT_IDLE_MAX | 8px | Max displacement per frame for "idle" |
| MOVEMENT_ACTIVE_MAX | 35px | Min displacement per frame for "walking" |
| RESTLESSNESS_HIGH_THRESHOLD | 12.0 | Std-dev pixels → restless |
| RESTLESSNESS_LOW_THRESHOLD | 2.0 | Std-dev pixels → frozen |
| IDLE_ALERT_SECONDS | 1800 | 30 minutes idle → prolonged_idle |
| TUMBLING_WINDOW_SEC | 7200 | 2-hour block |
| CLAHE clipLimit | 1.5 | Reduced from 2.0 to protect Re-ID colour fidelity |

---

## DATASETS RELEVANT TO THIS PROJECT

| Dataset | Use in pipeline |
|---|---|
| FER2013 (Kaggle) | Validate emotion model — NOT currently used (face removed) |
| COCO Keypoints | YOLOv8 and RTMPose were trained on this — underpins Stage 2 and 4 |

Future dataset consideration: build an Indian-workplace-specific posture annotation
dataset from actual CCTV footage using Roboflow or CVAT for annotation.

---

## ETHICAL CONSTRAINTS (hard requirements, never compromise)

- System is consent-based and fully opt-in
- Data never leaves the building — all processing on-premise
- Output is never used for performance review, appraisal, or enforcement
- Recommendations to managers are suggestive only, never prescriptive
- No facial data collected or stored
- Employees are informed they are part of the PoC

---

## WHAT IS DELIBERATELY NOT BUILT YET (future work)

1. **Dashboard UI** — A Streamlit or web frontend showing per-person index trends
   across the day. The session JSON log already contains all the data needed.

2. **India-specific posture dataset** — Annotate real CCTV footage with posture
   labels and fine-tune a YOLO classifier for Approach 2. Requires IRB approval,
   consent framework, and annotators.

3. **YOLO fine-tuning for posture** — Train YOLOv8 on annotated workplace CCTV
   specifically for posture class detection (sitting_upright, slouching, head_down,
   standing_active). This is Approach 2 of the pipeline.

4. **Multi-modal fusion** — Add audio stress cues (vocal pitch, speech rate) as
   an additional signal alongside posture.

5. **Cultural adaptation** — Non-verbal signals are culturally influenced.
   Indian workplace expressions of stress and disengagement may differ from
   Western datasets. Future dataset collection addresses this.

6. **Re-ID model fine-tuning** — OSNet pretrained on MSMT17 may benefit from
   fine-tuning on the specific clothing/appearance patterns of the target office.

---

## CODING CONVENTIONS FOR THIS PROJECT

- All thresholds and constants live in `config/settings.py` only.
  Never hardcode numbers inside logic files.
- Every function and class has a docstring explaining purpose and parameters.
- Use `get_logger(__name__)` from `utils/logger.py` in every module.
- Error handling: use try/except on all model inference calls. Return empty
  results rather than crashing the pipeline on a single bad frame.
- Stage modules return data in a consistent dict format. Do not change
  the dict schema of a stage without updating all downstream stages.
- No single file handles more than one concern. If a file is getting long,
  split it.
- Python type hints on all function signatures.

---

## HOW TO CONTINUE DEVELOPMENT

When asked to add a new feature or fix something, follow this checklist:

1. Does it require a new threshold? → Add to `config/settings.py` first.
2. Does it change a stage's output format? → Update all downstream stages.
3. Does it add a new wellness signal? → Add to `Observation` class in
   `temporal_fusion.py`, add weight to the relevant index in `settings.py`,
   add signal computation to `posture_classifier.py`.
4. Does it need testing without real footage? → Add a profile to
   `data/synthetic/profiles.py` and a generation case to `generator.py`.
5. Run syntax check: `python3 -c "import ast; ast.parse(open('file.py').read())"`.
6. After any change to classifier or fusion, re-run:
   `python -m data.synthetic.generator` and verify all 4 profiles still PASS.
