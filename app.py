"""
Streamlit app for flexibility assessment using pose detection
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Import pose utilities
from pose_utils import (
    mp_pose,
    assess_side_split,
    assess_posterior_chain,
    get_flexibility_rating,
    draw_pose_landmarks
)


class PoseProcessor(VideoProcessorBase):
    """Video processor for real-time pose detection with rolling average"""
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.max_split_angle = 0
        self.max_pc_angle = 180
        self.current_split_angle = 0
        self.current_pc_angle = 0
        
        # Rolling average buffers (last 5 frames)
        self.split_buffer = []
        self.pc_buffer = []
        self.buffer_size = 25
        
        # Track if we have valid readings
        self.is_reading_valid = False
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility > 0.5 and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility > 0.5 and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility > 0.5 and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.5:
            # Draw pose landmarks
            draw_pose_landmarks(img, results.pose_landmarks)
            
            # Get image dimensions
            h, w, _ = img.shape
            
            # Calculate raw metrics for this frame
            raw_split_angle = assess_side_split(
                results.pose_landmarks.landmark, w, h
            )
            raw_pc_angle = assess_posterior_chain(
                results.pose_landmarks.landmark, w, h
            )
            
            # Add to rolling buffers
            self.split_buffer.append(raw_split_angle)
            self.pc_buffer.append(raw_pc_angle)
            
            # Keep only last 5 frames
            if len(self.split_buffer) > self.buffer_size:
                self.split_buffer.pop(0)
            if len(self.pc_buffer) > self.buffer_size:
                self.pc_buffer.pop(0)
            
            # Only use readings if we have 5 consecutive frames
            if len(self.split_buffer) == self.buffer_size and len(self.pc_buffer) == self.buffer_size:
                self.is_reading_valid = True
                
                # Calculate rolling averages
                self.current_split_angle = np.mean(self.split_buffer)
                self.current_pc_angle = np.mean(self.pc_buffer)
                
                # Update max values
                if self.current_split_angle > self.max_split_angle:
                    self.max_split_angle = self.current_split_angle
                if self.current_pc_angle < self.max_pc_angle:
                    self.max_pc_angle = self.current_pc_angle
            else:
                self.is_reading_valid = False
            
            # Draw metrics on screen
            if self.is_reading_valid:
                cv2.putText(
                    img, f"Side Split: {self.current_split_angle:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    img, f"Max Split: {self.max_split_angle:.1f} deg",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                cv2.putText(
                    img, f"Post. Chain: {self.current_pc_angle:.1f} deg",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    img, f"Max PC: {self.max_pc_angle:.1f} deg",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
            else:
                # Show "calibrating" message
                frames_needed = self.buffer_size - len(self.split_buffer)
                cv2.putText(
                    img, f"Calibrating... ({frames_needed} frames needed)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2
                )
        else:
            # No pose detected - clear buffers
            self.split_buffer.clear()
            self.pc_buffer.clear()
            self.is_reading_valid = False
            
            cv2.putText(
                img, "No pose detected - ensure full body is visible",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def process_static_image(image_np):
    """Process a static image and display results"""
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image_rgb.shape
    
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Draw landmarks
        annotated_image = image_np.copy()
        draw_pose_landmarks(annotated_image, results.pose_landmarks)
        
        st.image(annotated_image, caption='Pose Detection', use_container_width=True)
        
        st.header("📊 Flexibility Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Side Split")
            split_angle = assess_side_split(
                results.pose_landmarks.landmark, image_width, image_height
            )
            rating, emoji = get_flexibility_rating("side_split", split_angle)
            
            st.metric("Leg Spread Angle", f"{split_angle:.1f}°")
            st.write(f"**Rating:** {rating} {emoji}")
            st.progress(min(split_angle / 180, 1.0))
            
        with col2:
            st.subheader("Posterior Chain")
            pc_angle = assess_posterior_chain(
                results.pose_landmarks.landmark, image_width, image_height
            )
            rating, emoji = get_flexibility_rating("posterior_chain", pc_angle)
            
            st.metric("Hip Flexion Angle", f"{pc_angle:.1f}°")
            st.write(f"**Rating:** {rating} {emoji}")
            st.progress(1.0 - min(pc_angle / 180, 1.0))
        
        st.info("""
        **How to interpret:**
        - **Side Split:** Higher angle = better flexibility (180° is full split)
        - **Posterior Chain:** Lower angle = better flexibility (0° means torso touching legs)
        """)
    else:
        st.error("⚠️ No pose detected in the image.")


def main():
    """Main Streamlit app"""
    st.title("💪 Flexibility Assessment Tool")
    st.write("Use real-time video or upload a photo to assess your flexibility!")
    
    # Sidebar instructions
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
    input_method = st.radio(
        "Choose input method:",
        ["🎥 Real-time Video", "📷 Take Photo", "📁 Upload Image"]
    )
    
    if input_method == "🎥 Real-time Video":
        st.subheader("Live Pose Detection")
        st.write("**Green values** = Current measurement | **Yellow values** = Best achieved")
        
        # RTC Configuration for WebRTC
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        ctx = webrtc_streamer(
            key="pose-detection",
            video_processor_factory=PoseProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if ctx.video_processor:
            st.info("📊 Your max scores are being tracked in real-time!")
            
            if st.button("🔄 Reset Max Scores"):
                if ctx.video_processor:
                    ctx.video_processor.max_split_angle = 0
                    ctx.video_processor.max_pc_angle = 180
                    ctx.video_processor.split_buffer.clear()
                    ctx.video_processor.pc_buffer.clear()
                    ctx.video_processor.is_reading_valid = False
                    st.success("Max scores reset!")
    
    elif input_method == "📷 Take Photo":
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            image_np = np.array(image)
            process_static_image(image_np)
    
    else:  # Upload Image
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            process_static_image(image_np)
    
    # Show example poses for static modes
    if input_method != "🎥 Real-time Video":
        st.subheader("Example Poses")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Side Split Pose**")
            st.write("Face the camera with legs spread wide")
        with col2:
            st.write("**Forward Fold Pose**")
            st.write("Stand sideways, bend forward reaching for toes")


if __name__ == "__main__":
    main()