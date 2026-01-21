# OpenVLA-OFT Perception Confusion Testing - Progress

## Project Goal
Test perception confusion in two-robot peg-in-hole task where:
- Robot A (robot0) carries the peg ("pegger")
- Robot B (robot1) carries the square with hole ("receiver")
- Hypothesis: Each robot seeing the other robot's body causes "perception confusion"
- Solution: Mask out robot bodies from each other's view

## Environment Setup

### Step 1: Virtual Environment (COMPLETED)
- Created Python 3.10 virtual environment using `uv`
- Location: `/workspace/openvla-oft/.venv`
- Activation: `source /workspace/openvla-oft/.venv/bin/activate`

### Step 2: Dependencies (COMPLETED)
- PyTorch 2.2.0 with CUDA 12.1 support
- All openvla-oft dependencies installed via `pip install -e .`
- Robosuite 1.5.2 installed for TwoArmPegInHole simulation
- imageio-ffmpeg for video output

### Step 3: HuggingFace Configuration (COMPLETED)
- HF cache directory: `/workspace/openvla-oft/.cache/huggingface`
- Model downloaded: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- Model location: `/workspace/openvla-oft/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial-object-goal-10/`

### Step 4: Headless Rendering Setup (COMPLETED)
- Using EGL for headless rendering
- Environment variables: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`

## Hardware
- 2x NVIDIA A100-SXM4-80GB GPUs
- CUDA Version: 12.8
- Driver Version: 570.195.03

## Segmentation Pipeline Explanation

### How Instance Segmentation Works in Robosuite

1. **Environment Creation**: When creating the robosuite environment with `camera_segmentations='instance'`, MuJoCo renders an additional segmentation image for each camera.

2. **Segmentation Output**: Each pixel in the segmentation image contains an integer ID corresponding to which object instance it belongs to:
   - ID 0: Background/table (~64% of pixels)
   - ID 1: Robot0 body - the pegger (~7% of pixels)
   - ID 2: Robot1 body - the receiver (~10% of pixels)
   - ID 4: Peg object (~8% of pixels)
   - ID 5: Hole object - square with hole (~10% of pixels)

3. **Masking Process**:
   - For Robot0's view: Create binary mask where `segmentation == 2` (Robot1)
   - Dilate mask using 7x7 kernel (3 iterations) for complete coverage
   - Apply inpainting to fill masked pixels naturally

4. **Visualization Colors** (in test video):
   - Gray = Background
   - Red = Robot0 (pegger)
   - Blue = Robot1 (receiver)
   - Green = Peg
   - Yellow = Hole

## Quick Start Commands

```bash
# Activate environment
cd /workspace/openvla-oft
source .venv/bin/activate

# Set environment variables
export HF_HOME=/workspace/openvla-oft/.cache/huggingface
export HF_TOKEN=<your_huggingface_token>
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Run masking demo (quick test)
python experiments/robot/twoarmpeginhole/test_masking.py

# Run perception experiments (baseline + masked)
python experiments/robot/twoarmpeginhole/run_perception_experiments.py
```

## Implementation Progress

### Robot Body Masking (COMPLETED)
- [x] Enable instance segmentation in robosuite
- [x] Extract robot body masks for each arm
- [x] Apply masks to remove other robot from each robot's view
- [x] Integrate with evaluation pipeline

### Files Created
- `Progress.md` - This file
- `experiments/robot/twoarmpeginhole/perception_masking.py` - Robot masking utilities
- `experiments/robot/twoarmpeginhole/test_masking.py` - Quick masking demo
- `experiments/robot/twoarmpeginhole/run_perception_experiments.py` - Full experiment runner

## Experiments

### Experiment 1: Baseline (No Masking)
- Both robots see each other fully
- Robot0 prompt: "insert the peg into the hole"
- Robot1 prompt: "align the hole to receive the peg"

### Experiment 2: With Masking
- Robot0 sees: itself + peg + hole (Robot1 body masked out)
- Robot1 sees: itself + peg + hole (Robot0 body masked out)
- Same prompts as baseline

## Output Locations
- Test videos: `./test_outputs/`
- Experiment videos: `./experiment_outputs/`

## Notes
- Model trained on LIBERO uses 7D actions (6D pose + 1D gripper)
- TwoArmPegInHole needs 12D actions (6D per robot, no gripper)
- Action chunks of 8 steps executed open-loop
- Actions scaled by 0.5 to adapt LIBERO model to TwoArm task
