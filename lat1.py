import cv2
import mediapipe as mp
import numpy as np
import joblib
import  pygame
import time

# Inisialisasi pose detection model dari MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi hand detection model dari MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Muat model klasifikasi gerakan vb yang telah dilatih
model = joblib.load('path_to_your_trained_model.pkl')

# Inisialisasi pygame untuk memutar suara
pygame.mixer.init()

# Daftar gerakan dan suara yang sesuai
actions = ["wave", "clap", "raise_hand"]  # Sesuaikan dengan gerakan yang telah dilatih
sounds = ["wave_sound.mp3", "clap_sound.mp3", "raise_hand_sound.mp3"]  # Path ke file suara


def draw_landmarks_with_labels(frame, landmarks, is_hand=False):
    """
    Draw landmarks and their labels on the frame.

    Parameters:
    - frame: The image frame on which to draw.
    - landmarks: The list of landmarks detected by MediaPipe.
    - is_hand: Boolean to check if the landmarks are from hand detection.
    """
    for i, landmark in enumerate(landmarks.landmark):
        # Get the coordinates of the landmark
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])

        # Draw a small circle at the landmark position
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        # Draw the index of the landmark near the circle
        cv2.putText(frame, f"H{i}" if is_hand else str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def predict_action(landmarks):
    """
    Predict the action based on pose landmarks.

    Parameters:
    - landmarks: The pose landmarks detected by MediaPipe.

    Returns:
    - action: The predicted action.
    """
    # Extract features from landmarks
    features = []
    for landmark in landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])

    # Convert to numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # Predict the action
    action_index = model.predict(features)[0]

    return actions[action_index]


def play_sound(action):
    """
    Play the sound corresponding to the action.

    Parameters:
    - action: The detected action.
    """
    try:
        sound_index = actions.index(action)
        sound_path = sounds[sound_index]
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
    except ValueError:
        print("Action not found in the actions list.")


def main():
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    last_action = None
    last_play_time = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print('Image not captured')
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect pose
        pose_result = pose.process(frame_rgb)
        # Process the frame to detect hands
        hands_result = hands.process(frame_rgb)

        # Create a copy of the original frame for displaying original frame without annotations
        frame_original = frame.copy()

        # Check if any pose landmarks are detected
        if pose_result.pose_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            # Draw landmarks with labels
            draw_landmarks_with_labels(frame, pose_result.pose_landmarks)

            # Predict action
            action = predict_action(pose_result.pose_landmarks)

            # Play sound if the action changes
            current_time = time.time()
            if action != last_action and current_time - last_play_time > 1:
                play_sound(action)
                last_action = action
                last_play_time = current_time
        else:
            print("No pose landmarks detected.")

        # Check if any hand landmarks are detected
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

                # Draw landmarks with labels
                draw_landmarks_with_labels(frame, hand_landmarks, is_hand=True)
        else:
            print("No hand landmarks detected.")

        # Display the original frame without annotations
        cv2.imshow("Original Frame", frame_original)

        # Display the frame with landmarks
        cv2.imshow("Pose and Hand Detection", frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
