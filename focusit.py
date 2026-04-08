import cv2 
import time
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# Initialize Google MediaPipe Face Mesh for facial landmark detection
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # This enables iris/eye tracking landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize timer variables starting from 0
focus_time_seconds = 0
lost_time_seconds = 0

# State variables to track what the system is currently doing
manual_pause = False # Tracks if the user pressed the shortcut key
last_tick_time = 0
is_first_frame = True # Flag to detect the very first camera frame

# Open the built-in webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    current_time = time.time()

    # STARTUP FIX
    # Prevent the 1-2 second jump at startup caused by the camera hardware warming up
    if is_first_frame:
        last_tick_time = current_time
        is_first_frame = False

    # Convert the BGR image to RGB before processing with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to find face and eye landmarks
    results = face_mesh.process(image_rgb)

    # KEYBOARD SHORTCUT
    # Listen for key presses (1ms delay to keep the video feed smooth)
    key = cv2.waitKey(1) & 0xFF
    
    # If the user presses 'ESC' (ASCII 27), close the program
    if key == 27:
        break
    # If the user presses 'p' (ASCII 112) or 'Spacebar' (ASCII 32), toggle manual pause
    elif key == 112 or key == 32:
        manual_pause = not manual_pause # Flips between True and False

    # TIMER
    if manual_pause:
        # If manually paused, lock the last_tick_time to the current_time.
        # This completely freezes BOTH timers and prevents background accumulation.
        last_tick_time = current_time
        status_text = "PAUSED (BREAK)"
        color = (0, 255, 255) # Yellow text
    else:
        # Check if AI detects the user's face/eyes
        if results.multi_face_landmarks:
            status_text = "FOCUSED"
            color = (0, 255, 0) # Green text
            is_focused = True
        else:
            status_text = "DISTRACTED"
            color = (0, 0, 255) # Red text
            is_focused = False

        # Update the appropriate counter exactly once per second
        if current_time - last_tick_time >= 1.0:
            if is_focused:
                focus_time_seconds += 1 # Count up the focused time
            else:
                lost_time_seconds += 1  # Count up the lost/distracted time
            
            last_tick_time = current_time

    # UI DISPLAY 
    # Format the seconds into MM:SS format using divmod (division and modulo)
    f_mins, f_secs = divmod(focus_time_seconds, 60)
    l_mins, l_secs = divmod(lost_time_seconds, 60)
    
    focus_text = f"Focus: {f_mins:02d}:{f_secs:02d}"
    lost_text = f"Lost : {l_mins:02d}:{l_secs:02d}"

    # Display the texts on the webcam feed
    # Focus time (Top)
    cv2.putText(image, focus_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    # Lost time (Middle)
    cv2.putText(image, lost_text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    # Status indicator (Bottom)
    cv2.putText(image, status_text, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show the final image
    cv2.imshow('Focusit - AI Powered Timer', image)

# Clean up and release the camera resources
cap.release()
cv2.destroyAllWindows()
