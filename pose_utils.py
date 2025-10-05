"""
Pose detection utilities for flexibility assessment
"""
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    
    Args:
        a: First point [x, y]
        b: Vertex point [x, y]
        c: Third point [x, y]
    
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def get_landmark_coords(landmarks, landmark_id, image_width, image_height):
    """
    Extract x, y coordinates from landmark
    
    Args:
        landmarks: MediaPipe pose landmarks
        landmark_id: ID of the landmark to extract
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        [x, y] coordinates
    """
    landmark = landmarks[landmark_id]
    return [landmark.x * image_width, landmark.y * image_height]


def assess_side_split(landmarks, image_width, image_height):
    """
    Assess side split angle
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Angle between legs in degrees (higher = better flexibility)
    """
    left_hip = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, image_width, image_height
    )
    right_hip = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, image_width, image_height
    )
    left_ankle = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, image_width, image_height
    )
    right_ankle = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, image_width, image_height
    )
    
    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    split_angle = calculate_angle(left_ankle, mid_hip, right_ankle)
    
    return split_angle


def assess_posterior_chain(landmarks, image_width, image_height):
    """
    Assess posterior chain flexibility (toe touch/forward fold)
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Hip flexion angle in degrees (lower = better flexibility)
    """
    left_shoulder = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, image_width, image_height
    )
    left_hip = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, image_width, image_height
    )
    left_ankle = get_landmark_coords(
        landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, image_width, image_height
    )
    
    hip_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
    
    return hip_angle


def get_flexibility_rating(metric, value):
    """
    Get flexibility rating based on metric and value
    
    Args:
        metric: Type of metric ('side_split' or 'posterior_chain')
        value: Angle value in degrees
    
    Returns:
        Tuple of (rating_text, emoji)
    """
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
    
    return "Unknown", "❓"


def draw_pose_landmarks(image, landmarks):
    """
    Draw pose landmarks on image
    
    Args:
        image: Image to draw on (numpy array)
        landmarks: MediaPipe pose landmarks
    
    Returns:
        Image with landmarks drawn
    """
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    return image