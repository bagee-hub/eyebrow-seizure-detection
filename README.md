# Eyebrow Detection App

A real-time eyebrow raise detection application using MediaPipe Face Mesh and OpenCV. Detects when you raise your left or right eyebrow and plays alert beep sounds.

---

## ⚠️ CRITICAL DISCLAIMER

**FOR DEMONSTRATION AND EDUCATIONAL PURPOSES ONLY**

This application is provided **AS-IS** for demonstration and learning purposes only. It is **NOT** intended for:

- ❌ Clinical diagnosis
- ❌ Medical decision-making
- ❌ Critical patient monitoring
- ❌ Life-safety applications
- ❌ Production medical environments

### Limitation of Liability

**THE DEVELOPER ASSUMES NO RESPONSIBILITY OR LIABILITY** for any consequences arising from the use of this program, including but not limited to:

- Medical misdiagnosis or missed diagnoses
- Delayed medical intervention
- False positive or false negative detections
- Technical failures or malfunctions
- Any harm or injury to persons
- Any loss or damage of any kind

**USE AT YOUR OWN RISK.** By using this software, you acknowledge and agree that:

1. This is an experimental educational tool only
2. It should never replace professional medical equipment or healthcare provider judgment
3. You are solely responsible for any consequences of its use
4. The developer provides no warranties of any kind, express or implied
5. This software has not been validated for medical use or approved by any regulatory authority

**Always consult qualified healthcare professionals for any medical concerns.**

---

## Medical Application

This application is designed to detect **partial complex seizures** in patients who experience involuntary eyebrow raises as a symptom. 

**Important Note**: Not all patients with partial complex seizures exhibit this symptom - only a subset of patients experience involuntary eyebrow raising during seizure episodes. This tool is specifically designed for those individuals who have been identified by their healthcare provider as having this particular manifestation.

Partial complex seizures (also known as focal impaired awareness seizures) can manifest with various motor symptoms, including:

- **Involuntary eyebrow raising**: Unilateral or bilateral elevation of eyebrows (occurs in a subset of patients)
- **Facial automatisms**: Repetitive facial movements
- **Altered awareness**: Patients may not be fully conscious during episodes

### Clinical Use Case

- **Background Monitoring**: Runs silently in headless mode while the patient works on their laptop
- **Real-time Alerts**: Provides immediate audio alerts when eyebrow raise patterns are detected
- **Non-intrusive**: Patient can continue normal computer work without interruption
- **Caregiver Notification**: Audio beeps alert nearby caregivers or the patient themselves of potential seizure activity
- **Documentation**: Helps track frequency and patterns of seizure events

### Platform Support

- **Tested Platform**: macOS (primary development and testing platform)
- **Experimental Support**: Linux and Windows (basic functionality should work but not extensively tested)

> **Medical Disclaimer**: This application is intended as a monitoring tool only and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for seizure management.

## Features

- **Real-time Detection**: Detects eyebrow raises using MediaPipe Face Mesh
- **Head Pose Compensation**: Handles head rotation up to ±25 degrees
- **Continuous Audio Alerts**: Beeps continuously while eyebrow is raised
- **Two Modes**: 
  - **Video Mode** (default): Shows live camera feed with visualization
  - **Headless Mode**: Runs in background without video window
- **Cross-platform Sound**: Works on macOS, Linux, and Windows
- **Toggleable Sound**: Enable/disable beep alerts on the fly

## Requirements

- Python 3.12 or higher
- Webcam/Camera access
- **macOS** (tested and recommended)
- Linux or Windows (experimental, not extensively tested)

## Installation

1. **Clone or download this repository**

2. **Create and activate virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Usage

### Video Mode (Default)

Shows live video feed with eyebrow detection visualization:

```bash
python eyebrow_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to toggle face landmarks visualization
- Press `m` to toggle sound on/off

### Headless Mode (Background)

Runs in background without showing video window (lower CPU usage):

```bash
python eyebrow_detection.py --headless
```

Or:

```bash
python eyebrow_detection.py --no-video
python eyebrow_detection.py --background
```

**Controls:**
- Press `Ctrl+C` to quit
- Type `m` + Enter to toggle sound on/off (Unix-like systems)

## File Structure

```
.
├── eyebrow_detection.py      # Main application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .venv/                     # Virtual environment (created locally)
└── captures/                  # Optional: for saving frames
```

## How It Works

1. **Face Detection**: Uses MediaPipe Face Mesh to detect 468 facial landmarks
2. **Eyebrow Tracking**: Monitors specific landmarks for left and right eyebrows:
   - Right eyebrow: landmarks [70, 63, 105, 66, 107]
   - Left eyebrow: landmarks [336, 296, 334, 293, 300]
3. **Head Pose Estimation**: Calculates head yaw angle to compensate for rotation
4. **Raise Detection**: Compares eyebrow heights relative to eyes
5. **Alert System**: Plays beep sound every 0.5 seconds while eyebrow is raised

## Customization

### Change Beep Interval

Modify the `beep_interval` variable in the main function:

```python
beep_interval = 15  # Beep every 15 frames (~0.5 sec at 30fps)
```

### Change Detection Threshold

Modify the `BASE_THRESHOLD` in the `detect_eyebrow_raise` method:

```python
BASE_THRESHOLD = 8  # Pixels difference threshold
```

### Change Sound File (macOS)

Modify the sound file path in `play_beep()`:

```python
subprocess.Popen(['afplay', '/System/Library/Sounds/Ping.aiff'], ...)
# Available sounds: Tink, Ping, Pop, Basso, Blow, Bottle, Frog, Funk, 
# Glass, Hero, Morse, Ping, Pop, Purr, Sosumi, Submarine, Tink
```

## Troubleshooting

### Camera Not Opening

- **Check permissions**: Grant camera access to Terminal/Python
  - macOS: System Settings → Privacy & Security → Camera
- **Camera in use**: Close other applications using the camera
- **Try different camera**: Modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### No Sound in Headless Mode

- Ensure your system sound is not muted
- On macOS, verify sound files exist: `ls /System/Library/Sounds/`
- On Linux, install `paplay`: `sudo apt-get install pulseaudio-utils`

### High CPU Usage

- Use headless mode for lower CPU usage
- Reduce camera resolution (modify `VideoCapture` settings)
- Increase beep interval to reduce processing

### Protobuf Warnings

These are suppressed automatically, but if you see them, they're harmless deprecation warnings from MediaPipe's internal libraries.

## Performance

- **Video Mode**: ~30 FPS with visualization and detection
- **Headless Mode**: ~30 FPS detection only (60-70% lower CPU usage)
- **Memory**: ~100-150 MB
- **Startup Time**: 2-3 seconds for MediaPipe initialization

## Dependencies

- **opencv-python**: Camera capture and image processing
- **mediapipe**: Face mesh detection and landmark tracking
- **numpy**: Numerical computations
- **Pillow**: Image handling

## Notes

- Works best with good lighting conditions
- Requires clear view of face and eyebrows
- Head rotation tolerance: ±25 degrees
- Beep sound plays asynchronously (non-blocking)

## Medical Monitoring Recommendations

### For Clinical Use:

1. **Lighting**: Ensure consistent, adequate lighting for reliable detection
2. **Camera Position**: Position camera to capture full face and eyebrows clearly
3. **Monitoring Duration**: Use headless mode for extended monitoring periods
4. **Alert Response**: Establish protocol for responding to beep alerts
5. **Documentation**: Log timestamps and frequency of detected events
6. **Calibration**: Adjust detection threshold based on patient's baseline eyebrow movement

### Limitations:

- **False Positives**: Normal voluntary eyebrow raises will trigger alerts
- **False Negatives**: May miss subtle or rapid movements
- **Environmental Factors**: Poor lighting or head position affects accuracy
- **Not Diagnostic**: Should be used as part of comprehensive seizure monitoring, not as sole diagnostic tool

### Privacy & Security:

- Video processing is done locally on device
- No data is transmitted or stored externally
- Captured frames are processed in real-time and not saved (unless explicitly configured)

## License

This project is provided under the MIT License for **educational and demonstration purposes only**.

### NO WARRANTY

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Medical Use Prohibition

This software is **NOT APPROVED, VALIDATED, OR INTENDED** for clinical, diagnostic, or critical medical use. Any use in medical contexts is strictly at the user's own risk, and the developer accepts no responsibility for any medical outcomes.

## Credits

- MediaPipe Face Mesh by Google
- OpenCV for computer vision
