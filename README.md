## linux setup guide
setup and activate venv and install requirements
```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

setup .env file in /yibuapi_examples_20260918_v01/.env
```bash
YIBU_API_KEY='your api key'
YIBU_AUDIT_LOG='artifacts/yibu_api_calls.jsonl'
```

run calibration for camera (on first run)
```bash
python start.py --camera 0 --port 8080 --calibrate
```

run camera without calibration (afterwards)
```bash
python start.py --camera 0 --port 8080
```

```
usage: start.py [-h] [--camera CAMERA] [--video VIDEO] [--host HOST]
                [--port PORT] [--fake] [--speed SPEED] [--calibrate]
                [--wake WAKE] [--chroma] [--no-embed] [--no-image]
                [--no-camera] [--no-voice] [--no-browser]

calibrate, then run the whole desk: world model, camera and voice

options:
  -h, --help       show this help message and exit
  --camera CAMERA  camera index, for calibration and perception
  --video VIDEO    run perception over a recording instead of the camera
  --host HOST
  --port PORT
  --fake           scripted observations instead of a camera; needs no
                   calibration
  --speed SPEED    --fake script speed
  --calibrate      recalibrate even if calib.json is already there
  --wake WAKE      say this, then the question
  --chroma         chroma detector instead of refdiff
  --no-embed       histograms instead of DINO
  --no-image       do not send the camera frame with a spoken question
  --no-camera      world model and voice only
  --no-voice       no microphone, no Omni voice calls
  --no-browser     do not open the dashboard
```
