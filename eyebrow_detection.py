"""
Eyebrow Raise Detection Demo
This script demonstrates how to detect if one eyebrow is raised over the other
"""

import warnings
# Suppress protobuf deprecation warnings from MediaPipe
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import numpy as np
import platform
import subprocess
import argparse
import sys

def play_beep(frequency=1000, duration=100):
    """
    Play a beep sound (cross-platform) - non-blocking
    
    Args:
        frequency: Frequency in Hz (only used on macOS)
        duration: Duration in milliseconds
    """
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # Use afplay with a generated tone - run in background without waiting
            subprocess.Popen(['afplay', '/System/Library/Sounds/Tink.aiff'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        elif system == "Linux":
            # Use beep command or paplay - run in background
            subprocess.Popen(['paplay', '/usr/share/sounds/freedesktop/stereo/bell.oga'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        elif system == "Windows":
            import winsound
            # winsound.Beep is synchronous, but fast enough on Windows
            winsound.Beep(frequency, duration)
    except Exception:
        # Fallback: print bell character (works on most terminals)
        print('\a', end='', flush=True)

class EyebrowDetector:
    """Detects and analyzes eyebrow positions using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eyebrows
        # Right eyebrow (from face perspective - left side of screen when mirrored)
        self.RIGHT_EYEBROW = [70, 63, 105, 66, 107]
        # Left eyebrow (from face perspective - right side of screen when mirrored)
        self.LEFT_EYEBROW = [336, 296, 334, 293, 300]
        

    def estimate_head_pose(self, landmarks, h, w):
        """
        Estimate head rotation angle (yaw) to compensate for head turning
        
        Returns:
            yaw_angle: Head rotation in degrees (-90 to +90)
                      Negative = turned left, Positive = turned right
        """
        # Key landmarks for pose estimation
        nose_tip = landmarks.landmark[1]
        left_eye_corner = landmarks.landmark[33]   # Left eye inner corner
        right_eye_corner = landmarks.landmark[263] # Right eye inner corner
        
        # Convert to pixel coordinates
        nose_x = nose_tip.x * w
        left_eye_x = left_eye_corner.x * w
        right_eye_x = right_eye_corner.x * w
        
        # Calculate face width at eye level
        eye_distance = abs(right_eye_x - left_eye_x)
        
        # Calculate nose position relative to eye center
        eye_center_x = (left_eye_x + right_eye_x) / 2
        nose_offset = nose_x - eye_center_x
        
        # Estimate yaw angle based on nose offset
        # Normalize by eye distance to account for face size
        if eye_distance > 0:
            normalized_offset = nose_offset / eye_distance
            # Convert to degrees (approximate)
            yaw_angle = normalized_offset * 45  # Rough calibration
        else:
            yaw_angle = 0
        
        return yaw_angle
    
    def detect_eyebrow_raise(self, frame):
        """
        Detects if one eyebrow is raised higher than the other
        Now with head pose compensation for rotation tolerance!
        
        Args:
            frame: BGR image from camera
            
        Returns:
            tuple: (status, difference, left_height, right_height, landmarks, yaw)
                - status: "no_face", "neutral", "right_raised", "left_raised", or "turn_forward"
                - difference: Vertical difference between eyebrows (pixels)
                - left_height: Height of left eyebrow relative to nose
                - right_height: Height of right eyebrow relative to nose
                - landmarks: Face landmarks object (for visualization)
                - yaw: Head rotation angle in degrees
        """
        # Convert to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return "no_face", 0, 0, 0, None, 0
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Estimate head pose
        yaw_angle = self.estimate_head_pose(face_landmarks, h, w)
        
        # Check if head is turned too far
        MAX_YAW_ANGLE = 25  # degrees
        if abs(yaw_angle) > MAX_YAW_ANGLE:
            return "turn_forward", 0, 0, 0, face_landmarks, yaw_angle
        
        # Extract eyebrow landmarks with 3D coordinates
        right_eyebrow_points_y = []
        right_eyebrow_points_z = []
        left_eyebrow_points_y = []
        left_eyebrow_points_z = []
        
        for idx in self.RIGHT_EYEBROW:
            landmark = face_landmarks.landmark[idx]
            right_eyebrow_points_y.append(landmark.y * h)
            right_eyebrow_points_z.append(landmark.z)  # Depth information
        
        for idx in self.LEFT_EYEBROW:
            landmark = face_landmarks.landmark[idx]
            left_eyebrow_points_y.append(landmark.y * h)
            left_eyebrow_points_z.append(landmark.z)
        
        # Calculate average heights (lower y = higher on screen)
        right_avg = np.mean(right_eyebrow_points_y)
        left_avg = np.mean(left_eyebrow_points_y)
        
        # Get reference points for better normalization
        nose_bridge = face_landmarks.landmark[168]
        nose_y = nose_bridge.y * h
        
        # Get eye landmarks for additional reference
        right_eye = face_landmarks.landmark[33]
        left_eye = face_landmarks.landmark[263]
        right_eye_y = right_eye.y * h
        left_eye_y = left_eye.y * h
        
        # Calculate eyebrow heights relative to corresponding eyes (better for rotation)
        right_height = right_eye_y - right_avg  # Distance from right eye to right eyebrow
        left_height = left_eye_y - left_avg     # Distance from left eye to left eyebrow
        
        # Apply compensation for head rotation
        # When head turns, perspective changes - adjust threshold
        rotation_compensation = abs(yaw_angle) / 10.0  # Increase tolerance when rotated
        
        # Calculate difference
        difference = abs(right_height - left_height)
        
        # Adaptive threshold based on head rotation
        BASE_THRESHOLD = 8
        RAISE_THRESHOLD = BASE_THRESHOLD + rotation_compensation
        
        # Determine status
        if difference < RAISE_THRESHOLD:
            status = "neutral"
        elif right_height > left_height:
            status = "right_raised"
        else:
            status = "left_raised"
        
        return status, difference, left_height, right_height, face_landmarks, yaw_angle
    
    def visualize_eyebrows(self, frame, show_all_landmarks=False):
        """
        Draw eyebrow landmarks and detection result on the frame
        
        Args:
            frame: BGR image from camera
            show_all_landmarks: If True, show all 468 face landmarks
            
        Returns:
            Annotated frame with eyebrow detection visualization
        """

        # Get detection results (now includes yaw angle)
        status, diff, left_h, right_h, landmarks, yaw = self.detect_eyebrow_raise(frame)
        
        if landmarks is None:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        h, w = frame.shape[:2]
        
        # Draw right eyebrow (RED)
        for idx in self.RIGHT_EYEBROW:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        # Draw left eyebrow (BLUE)
        for idx in self.LEFT_EYEBROW:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        
        # Connect eyebrow points with lines
        for i in range(len(self.RIGHT_EYEBROW) - 1):
            pt1_idx = self.RIGHT_EYEBROW[i]
            pt2_idx = self.RIGHT_EYEBROW[i + 1]
            pt1 = landmarks.landmark[pt1_idx]
            pt2 = landmarks.landmark[pt2_idx]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for i in range(len(self.LEFT_EYEBROW) - 1):
            pt1_idx = self.LEFT_EYEBROW[i]
            pt2_idx = self.LEFT_EYEBROW[i + 1]
            pt1 = landmarks.landmark[pt1_idx]
            pt2 = landmarks.landmark[pt2_idx]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Optionally show all face landmarks
        if show_all_landmarks:
            for landmark in landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        

        # Display result text with LARGE font for status
        status_text = status.replace('_', ' ').title()
        
        # Choose color based on status
        if status == "neutral":
            color = (0, 255, 0)  # Green
        elif status == "turn_forward":
            color = (0, 255, 255)  # Yellow - warning to turn forward
        else:
            color = (0, 165, 255)  # Orange - eyebrow raised
        
        # Very large status text in the center-top
        font_scale_status = 2.5  # Much larger font
        thickness_status = 6
        
        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale_status, thickness_status
        )
        
        # Position at top center
        text_x = (w - text_width) // 2
        text_y = 80
        
        # Draw background rectangle for better visibility
        padding = 20
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw the large status text
        cv2.putText(frame, status_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_DUPLEX, font_scale_status, color, thickness_status)
        

        # Smaller supporting information at the top left
        # Only show detailed info if not warning to turn forward
        if status != "turn_forward":
            cv2.putText(frame, f"Head Yaw: {yaw:.1f}°", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Diff: {diff:.1f}px", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"L: {left_h:.1f}px", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"R: {right_h:.1f}px", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Just show rotation angle when head is turned too far
            cv2.putText(frame, f"Head Yaw: {yaw:.1f}° (max ±25°)", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Legend
        cv2.putText(frame, "Red = Right Eyebrow | Blue = Left Eyebrow", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def main():
    """Main function to run the eyebrow detection demo"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Eyebrow Raise Detection Demo')
    parser.add_argument('--headless', '--no-video', '--background', 
                       action='store_true',
                       dest='headless',
                       help='Run in headless mode (no video window, only beeps)')
    args = parser.parse_args()
    
    print("Eyebrow Raise Detection Demo")
    print("=" * 50)
    if args.headless:
        print("MODE: HEADLESS (Background mode - no video window)")
        print("Instructions:")
        print("  - Camera is running in background")
        print("  - Beep sounds when eyebrow is raised")
        print("  - Press Ctrl+C to quit")
        print("  - Type 'm' + Enter to toggle sound")
    else:
        print("MODE: VIDEO (Display mode)")
        print("Instructions:")
        print("  - Position your face in front of the camera")
        print("  - Try raising one eyebrow at a time")
        print("  - A beep will sound when an eyebrow is raised!")
        print("  - Press 'q' to quit")
        print("  - Press 's' to toggle all landmarks")
        print("  - Press 'm' to toggle sound on/off")
    print("=" * 50)
    

    # Initialize detector
    print("Initializing detector... (this may take a few seconds)")
    try:
        detector = EyebrowDetector()
        print("Detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    

    show_all_landmarks = False
    sound_enabled = True
    frame_count = 0
    
    # Beep tracking - beep continuously while raised
    last_beep_frame = -100  # Frame number of last beep
    beep_interval = 15  # Beep every 15 frames (about 0.5 seconds at 30fps)
    last_status = "neutral"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Print status every 30 frames (about once per second at 30fps) - only in headless mode
            if args.headless and frame_count % 30 == 0:
                print(f"Processing frame {frame_count}... Status: {last_status}")
            
            try:
                # Mirror the frame for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Get detection status
                status, diff, left_h, right_h, landmarks, yaw = detector.detect_eyebrow_raise(frame)
                
                # Continuous beep logic - beep while eyebrow is raised
                if sound_enabled and status in ["right_raised", "left_raised"]:
                    # Beep continuously at regular intervals while raised
                    if frame_count - last_beep_frame >= beep_interval:
                        print(f"🔔 BEEP! {status.replace('_', ' ').title()}")
                        play_beep()
                        last_beep_frame = frame_count
                elif status == "neutral" and last_status in ["right_raised", "left_raised"]:
                    # Status returned to neutral
                    if args.headless:
                        print("✓ Back to neutral - beeping stopped")
                
                last_status = status
                
                # Only visualize and show window if NOT in headless mode
                if not args.headless:
                    # Detect and visualize eyebrow raise
                    frame = detector.visualize_eyebrows(frame, show_all_landmarks)
                    
                    # Add sound status indicator
                    sound_text = "🔊 Sound: ON" if sound_enabled else "🔇 Sound: OFF"
                    cv2.putText(frame, sound_text, (frame.shape[1] - 200, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if sound_enabled else (128, 128, 128), 2)
                    
                    # Display the frame
                    cv2.imshow('Eyebrow Raise Detection', frame)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Continue to next frame
                pass
            
            # Handle keyboard input
            if not args.headless:
                # Video mode - use cv2.waitKey for instant key detection
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    show_all_landmarks = not show_all_landmarks
                    print(f"Show all landmarks: {show_all_landmarks}")
                elif key == ord('m'):
                    sound_enabled = not sound_enabled
                    print(f"Sound {'enabled' if sound_enabled else 'disabled'}")
            else:
                # Headless mode - use non-blocking input check
                # Check if user input is available (non-blocking)
                import select
                if sys.platform != 'win32':
                    # Unix-like systems
                    if select.select([sys.stdin], [], [], 0)[0]:
                        user_input = sys.stdin.readline().strip().lower()
                        if user_input == 'q':
                            print("Quitting...")
                            break
                        elif user_input == 'm':
                            sound_enabled = not sound_enabled
                            print(f"Sound {'enabled' if sound_enabled else 'disabled'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Demo ended")

if __name__ == "__main__":
    main()
