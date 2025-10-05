import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def get_landmark_coords(landmarks, landmark_id, image_width, image_height):
    """Extract x, y coordinates from landmark"""
    landmark = landmarks[landmark_id]
    return [landmark.x * image_width, landmark.y * image_height]

def assess_side_split(landmarks, image_width, image_height):
    """Assess side split angle"""
    # Get hip, knee, and ankle landmarks for both legs
    left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, image_width, image_height)
    right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, image_width, image_height)
    left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, image_width, image_height)
    right_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, image_width, image_height)
    
    # Calculate the angle at the hips (between left leg and right leg)
    mid_hip = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
    split_angle = calculate_angle(left_ankle, mid_hip, right_ankle)
    
    return split_angle

def assess_posterior_chain(landmarks, image_width, image_height):
    """Assess posterior chain flexibility (toe touch/forward fold)"""
    # Get shoulder, hip, and ankle landmarks
    left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, image_width, image_height)
    left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, image_width, image_height)
    left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, image_width, image_height)
    
    # Calculate hip flexion angle
    hip_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
    
    return hip_angle

def get_flexibility_rating(metric, value):
    """Get flexibility rating based on metric and value"""
    if metric == "side_split":
        if value >= 160:
            return "Excellent", "🌟"
        elif value >= 140:
            return "Good", "👍"
        elif value >= 120:
            return "Fair", "💪"
        else:
            return "Needs Work", "📈"
    
    elif metric == "posterior_chain":
        if value <= 30:
            return "Excellent", "🌟"
        elif value <= 50:
            return "Good", "👍"
        elif value <= 70:
            return "Fair", "💪"
        else:
            return "Needs Work", "📈"

# Streamlit UI
st.title("💪 Flexibility Assessment Tool")
st.write("Upload a photo to assess your side split and posterior chain flexibility!")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
### Side Split Assessment
- Stand with legs spread as wide as possible
- Keep your body upright and facing the camera
- Ensure full body is visible

### Posterior Chain Assessment  
- Perform a forward fold (toe touch)
- Stand sideways to the camera
- Keep legs straight
- Reach towards your toes
""")

# Choose input method
input_method = st.radio("Choose input method:", ["📷 Take Photo with Webcam", "📁 Upload Image"])

image_np = None

if input_method == "📷 Take Photo with Webcam":
    camera_photo = st.camera_input("Take a picture")
    
    if camera_photo is not None:
        # Read image from camera
        image = Image.open(camera_photo)
        image_np = np.array(image)
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

if image_np is not None:
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    image_height, image_width, _ = image_rgb.shape
    
    # Process image with MediaPipe
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Draw pose landmarks on image
        annotated_image = image_np.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # Display annotated image
        st.image(annotated_image, caption='Pose Detection', use_container_width=True)
        
        # Calculate metrics
        st.header("📊 Flexibility Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Side Split")
            split_angle = assess_side_split(results.pose_landmarks.landmark, image_width, image_height)
            rating, emoji = get_flexibility_rating("side_split", split_angle)
            
            st.metric("Leg Spread Angle", f"{split_angle:.1f}°")
            st.write(f"**Rating:** {rating} {emoji}")
            st.progress(min(split_angle / 180, 1.0))
            
        with col2:
            st.subheader("Posterior Chain")
            pc_angle = assess_posterior_chain(results.pose_landmarks.landmark, image_width, image_height)
            rating, emoji = get_flexibility_rating("posterior_chain", pc_angle)
            
            st.metric("Hip Flexion Angle", f"{pc_angle:.1f}°")
            st.write(f"**Rating:** {rating} {emoji}")
            st.progress(1.0 - min(pc_angle / 180, 1.0))
        
        # Additional info
        st.info("""
        **How to interpret:**
        - **Side Split:** Higher angle = better flexibility (180° is full split)
        - **Posterior Chain:** Lower angle = better flexibility (0° means torso touching legs)
        """)
        
    else:
        st.error("⚠️ No pose detected in the image. Please ensure your full body is visible and try again.")

else:
    st.info("👆 Upload an image to get started!")
    
    # Show example layout
    st.subheader("Example Poses")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Side Split Pose**")
        st.write("Face the camera with legs spread wide")
    with col2:
        st.write("**Forward Fold Pose**")
        st.write("Stand sideways, bend forward reaching for toes")