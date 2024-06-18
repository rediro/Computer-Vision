import numpy as np
import cv2

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f_right, f_left):
    # Convert focal lengths from [mm] to [pixel]

    f_pixel_right = f_right / 3.45
    f_pixel_left = f_left / 3.45

    # Calculate disparity
    disparity = left_point[0] - right_point[0]

    # Calculate depth
    if disparity != 0:
        z_depth = 5.797101449275361*(baseline * f_pixel_right) / disparity
    else:
        z_depth = float('inf')  # Handle division by zero

    return z_depth

# Example camera parameters (replace with actual calibration results)
baseline = 20  # Baseline distance between cameras in [cm]
focal_length_right = 50  # Focal length of the right camera in [mm]
focal_length_left = 24  # Focal length of the left camera in [mm]

# Example corresponding points (replace with actual image points)
right_point = [100, 200]  # Example point in the right image
left_point = [150, 200]   # Corresponding point in the left image

# Example frames (replace with actual frames from camera)
frame_right = np.zeros((1440,900 , 3), dtype=np.uint8)  # Example right frame
frame_left = np.zeros(( 2778 , 1284, 3), dtype=np.uint8)   # Example left frame

# Calculate depth
depth = find_depth(right_point, left_point, frame_right, frame_left, baseline, focal_length_right, focal_length_left)
print("Depth:", depth)
