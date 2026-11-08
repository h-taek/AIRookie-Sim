# AIRookie_Sim

> Simulator app for AI Rookie, a small bipedal companion robot. It runs on a Mac, and the real onboard software runs inside it.

English | [한국어](README.ko.md)

## What it solves

The autonomy software (call, search, follow, emotion, expression) had to be built before the real robot existed, and earlier checks skipped the actual pipeline by feeding precomputed results. In this app the simulator plugs in where the servos, IMU, camera and microphone plug in on the robot. The onboard software does not know whether it runs in simulation or on hardware, so detection, transcription and judgment all run for real.

## Who it is for

- People who want to test an autonomy stack end to end before the hardware is ready
- Anyone building a MuJoCo app with custom widgets around a learned walking policy

## Features

- MuJoCo world with the training robot model (servo backlash included) and a movable 3D person
- Robot camera view and spectator view side by side, with live readouts from the onboard software
- Two verification modes: following (world render as the robot camera) and emotion expression (speech face clip or MacBook camera)
- Sound input from a speech WAV, typed text read by macOS TTS, or the MacBook microphone
- Policy switching from the panel, and a real-robot mode that launches the Jetson entry point and shows its camera and readouts in the same window

## How it works

- One cycle is 20 ms. `bridge.py` turns the world state into the sensor bundle defined by the driver contract, calls the onboard software, and applies its commands back to the world.
- The onboard software is attached by path, not installed. `pyproject.toml` sets `pythonpath = ["src", "../AIRookie_Runtime/src"]`, so the three repositories sit side by side.

| Module | Role |
|---|---|
| `world.py` | Physics world: robot, person, initial pose |
| `bridge.py` | World to sensor bundle, commands to world |
| `views.py` | Robot camera and spectator renders |
| `sources.py` · `inject.py` | Video and sound sources, external injection |
| `panels.py` · `app.py` | Readout selection and window drawing (dearpygui) |

## Requirements and usage

- macOS, Python 3.11 or later, [uv](https://docs.astral.sh/uv/)
- [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) and [AIRookie_Playground](https://github.com/h-taek/AIRookie-Playground) cloned next to this repository

```bash
uv venv
uv pip install mujoco dearpygui numpy onnxruntime openai-whisper pytest
uv pip install opencv-python-headless sounddevice      # MacBook camera, microphone, speech faces
```

Export the person detector once from the onboard repository:

```bash
cd ../AIRookie_Runtime
uv run --no-project --with ultralytics --with onnx --with onnxslim python tools/export_detector.py
```

Run:

```bash
.venv/bin/python run.py                         # default policy v2_48
.venv/bin/python run.py --policy <policy.onnx>
.venv/bin/python run.py --real                  # connect to the real robot (ssh alias "jetson")
.venv/bin/python -m pytest -q                   # tests that do not touch devices
.venv/bin/python -m pytest -q -m device         # microphone, camera, TTS
```

## Data

The person mesh and emotion clips are not in this repository. Prepare the mesh with `tools/prepare_person_mesh.py <mesh.obj>`. The emotion clips are cut from the AI Hub dataset "한국인 감정인식을 위한 복합영상" (multimodal video for Korean emotion recognition), whose terms do not allow redistribution, so each user obtains them from AI Hub and places them in `assets/emotion_video/`.

## Related repositories

| Repository | Role |
|---|---|
| [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) | Onboard software run inside this app |
| [AIRookie_Playground](https://github.com/h-taek/AIRookie-Playground) | Walking policies and robot model |

## License

No open-source license has been chosen yet; all rights reserved.
