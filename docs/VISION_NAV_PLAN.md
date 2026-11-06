# Vision-based autonomous navigation — plan

Goal: make Eshy navigate and avoid obstacles autonomously using its onboard
camera, with a real ML/RL learning component (not hand-coded rules).

The robot streams MJPEG over WiFi to the laptop, the laptop runs perception
+ policy, and sends discrete high-level commands back to the robot over BLE
that trigger the existing trained walk / turn / stop skills.

---

## Architecture: pretrained perception → structured state → small learned policy

```
                    ┌─► YOLOv8n (object detection)
                    │      ↓
                    │   list of (class, bbox, confidence)
camera frame ──────►│
                    │   per-pixel depth map
                    │      ↑
                    └─► Depth Anything V2 Small (monocular depth)

                                ↓
                    structured state vector
                    (≈25 floats: top-K detections + free-space features)

                                ↓
                    learned MLP policy
                                ↓
                    {forward, left, right, stop}
                                ↓
                    BLE command to robot ─► existing walk / turn / stand skills
```

### Why this design

| | Raw pixels → action | This: pretrained perception → state → action |
|---|---|---|
| Data needed | ~50K frames | ~1–5K frames |
| Network size | CNN backbone (~5M params) | tiny MLP (~10K params) |
| Generalization to new rooms | poor without lots of data | good (state vector looks the same in any room) |
| Interpretability | "the model decided" | "I see chair @ 0.5m to right → turn left" |
| Sim-to-real for vision | brutal | trivial (perception works the same) |
| RL feasibility | very hard | tractable |

Trade-off: any signal not captured by the perception models is invisible to
the policy. COCO classes cover ~80 common obstacles; for the rest, the
depth-map free-space features carry most of the load.

---

## Pretrained models

### Object detection: YOLOv8n
- `pip install ultralytics`
- `~30 fps` on a laptop CPU
- COCO classes: chair, couch, table, person, dog, cat, refrigerator, etc.
- Start here. Upgrade to YOLOv11n later if you want.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict(frame)
```

### Depth: Depth Anything V2 Small
- Public weights via Hugging Face (`depth-anything/Depth-Anything-V2-Small-hf`)
- ~30 fps on GPU, ~5 fps on CPU
- Produces *relative* depth — calibrate scale once with a known-distance
  reference object
- For metric depth out of the box, use **ZoeDepth** instead (slower)

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
```

If running both models simultaneously is too slow on CPU, drop inference
to 5 fps — fine for a walking quadruped — or pick up a small GPU.

---

## State vector design

```
For the K=3 nearest detections (sorted by distance):
  class_id          (one-hot or learned embedding)
  distance          (normalized 0..1 from the calibrated depth scale)
  bearing           (-1 = left edge, 0 = center, +1 = right edge; from bbox center x)
  bbox_area_pct     (proxy for size / closeness)

Global free-space features (from the depth map):
  min_depth_left_third     (nearest obstacle in left third of view)
  min_depth_center_third
  min_depth_right_third
  mean_depth_below_horizon (floor visibility / drop-offs)
```

≈25 floats total. A 2-layer MLP with 64 hidden units handles this trivially.

The state design will iterate. Plan to revise it once you see what fails
in real episodes — that's part of the practice.

---

## Policy training options

### Option 1: Behavior cloning (recommended start)
- Drive the robot manually with the keyboard, log `(state_vector, action_label)` pairs
- ~5–10 minutes of data is enough — the state space is small, the learning task is simple
- Train a tiny MLP with cross-entropy loss, ~1 minute on CPU
- Most feasible starting point. Get a working policy in one weekend.

### Option 2: Real-world RL
With a 25-dim state space and a 4-class discrete action space, online RL
becomes tractable:
- Reward: `+forward_progress − collision_penalty − time_penalty`
- PPO with stable-baselines3 (already in the project)
- ~30–60 min of robot time should produce a working policy

### Option 3: Hand-rule + ML refinement
- Bootstrap with a simple rule policy ("turn toward the third with the largest min_depth")
- Run it to collect data + apply human safety overrides
- Train MLP to imitate the rule + corrections — the MLP learns the corner cases
the rule misses

---

## Phase roadmap

| Phase | What | Time |
|---|---|---|
| **0** | Camera streaming + manual teleop + data logging end-to-end | 1 weekend |
| **1** | Wire up YOLO + Depth Anything; visualize the state vector live | 1 weekend |
| **2** | BC training pipeline; drive ~10 min; train MLP; deploy on robot | 1 weekend |
| **3** | Iterate state design; collect more data; retrain | 1 weekend |
| **4** | (optional) Real-world RL fine-tuning | 1–2 weekends |
| **5** | (optional) Goal conditioning ("go to the chair") | 1 weekend |
| **6** | (optional) Memory / LSTM head for stuck-in-loop avoidance | 1 weekend |

Core working demo: **3–4 weekends**.
Optional advanced extensions: another 3–4 weekends.

---

## Caveats / honest gotchas

1. **Two pretrained models in parallel is slow on CPU.** Run inference at 5 fps
   (fine for a slow quadruped) or use a small GPU.
2. **Depth Anything is relative.** Calibrate scale once with a known-distance
   reference object. Or use ZoeDepth for metric out of the box.
3. **YOLO misses unusual obstacles.** Table legs, wires, low rugs may not get
   detected. The depth-map free-space features are the safety net for this.
4. **State design is iterative.** Your first state vector won't be the right
   one. Expect to revise after seeing real failures.
5. **Pretrained models fail badly sometimes** — glossy surfaces break depth,
   weird textures cause false detections. Add sanity checks (e.g., depth
   shouldn't change by 5m between consecutive frames).
6. **Latency**: BLE + WiFi + perception adds ~150–300 ms total round-trip.
   Fine for slow walking; not for fast reactive control.

---

## Why this is the most feasible ML path

- **No simulation required.** Sim-to-real for vision is the hardest part of any
  learned-vision project. Pretrained perception sidesteps it.
- **Tiny dataset works.** The hard recognition has already been learned by the
  pretrained models on millions of frames; you only train the small policy.
- **Builds on what already exists.** Walk and turn skills are trained — the
  vision policy just selects which one to call.
- **Real ML practice.** Imitation learning, possibly RL, structured-state
  policies, hierarchical control — all real techniques used in production
  robotics.
- **Honest demo.** A robot wandering a room avoiding furniture using a CNN +
  depth model + learned policy is a strong portfolio piece. It's also the
  approximate level of where small-quadruped robotics research was in ~2018,
  achievable on hobby hardware in 2025 because of mature pretrained models.

---

## What I'm NOT doing (and why)

- **Visual SLAM (ORB-SLAM3 etc.)** — fragile, classical CV, not really ML
  practice. AprilTag + odometry is the lighter cheaper alternative if I ever
  need a global map.
- **End-to-end vision-RL in simulation** — sim-to-real for vision is multiple
  weekends of domain randomization for arguable benefit at hobby scale.
- **Pretrained navigation policies (GNM / NoMaD / Habitat agents)** —
  action-space mismatch (their robots are wheeled). Bridging it is more work
  than just doing BC from my own demos with a pretrained encoder.
- **Pretraining on extra navigation datasets (RECON / Gibson / Ego4D)** —
  diminishing returns at hobby scale once you have a strong pretrained
  perception stack and your own demonstrations.

---

## First concrete deliverable (perception sandbox)

Before touching the robot side, get a Python script working on the laptop that:

1. Captures a frame from the camera stream (or a test image).
2. Runs YOLOv8n → list of detections.
3. Runs Depth Anything V2 Small → depth map.
4. Computes the state vector.
5. Visualizes it live (overlay bboxes + depth heatmap + state values).

This unblocks Phase 1 and lets you iterate on perception while the rest of
the hardware is being assembled.
