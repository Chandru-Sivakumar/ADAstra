
import pyzed.sl as sl
import numpy as np
import cv2

# Initialize Camera
def init_camera():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Highest accuracy
    init_params.coordinate_units = sl.UNIT.METER  # Output in meters
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        exit(1)
    return zed


# Read Left & Right Stereo Images
def get_stereo_images(zed):
    left_image = sl.Mat()
    right_image = sl.Mat()
    
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(left_image, sl.VIEW.LEFT)  
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        
    return left_image.get_data(), right_image.get_data()


## Read Depth Map within Detected Bounding Box
def get_depth_map_bbox(zed, detections):
    depth_map = sl.Mat()
    
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

    depth_data = depth_map.get_data()
    depths = {}

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det)  # Extract bounding box and class
        
        # Extract depth data within bounding box
        bbox_depth = depth_data[y1:y2, x1:x2]
        
        # Avoid invalid depth values (e.g., NaN or extreme values)
        valid_depths = bbox_depth[np.isfinite(bbox_depth)]
        
        if valid_depths.size > 0:
            depths[cls] = np.mean(valid_depths)  # Store average depth per class

    return depths if depths else None  # Return dictionary with class-wise depth data



# Read Camera Pose
def get_camera_pose(zed):
    pose = sl.Pose()

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    
    translation = pose.get_translation().get()
    rotation = pose.get_rotation_matrix().r
    
    return {
        "Translation": translation,
        "Rotation Matrix": rotation
    }


# Read IMU Sensor Data
def get_imu_data(zed):
    sensors_data = sl.SensorsData()
    
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)
    
    acceleration = sensors_data.get_accelerometer_data().get_linear_acceleration()
    gyroscope = sensors_data.get_gyroscope_data().get_angular_velocity()
    
    return {
        "Acceleration": acceleration,
        "Gyroscope": gyroscope
    }

# Overlay IMU data on the image
def overlay_imu_data(image, imu_data):
    overlay = image.copy()
    
    # Convert IMU values to string
    acc_text = f"Acc: x={imu_data['Acceleration'][0]:.2f}, y={imu_data['Acceleration'][1]:.2f}, z={imu_data['Acceleration'][2]:.2f}"
    gyro_text = f"Gyro: x={imu_data['Gyroscope'][0]:.2f}, y={imu_data['Gyroscope'][1]:.2f}, z={imu_data['Gyroscope'][2]:.2f}"
    
    # Draw text on image
    cv2.putText(overlay, acc_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, gyro_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return overlay

def display_live_feed(zed, detections):
    while True:
        # Capture stereo images
        left_img, right_img = get_stereo_images(zed)

        # Get depth data for detected objects
        depth_values, depth_map = get_depth_map_bbox(zed, detections)

        # Convert images for OpenCV
        left_img_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGR)
        right_img_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGBA2BGR)

        # Normalize depth map for visualization
        depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(depth_map_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # Overlay detection results and depth values
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det)
            color = (0, 255, 0)  # Green box for detected objects
            cv2.rectangle(left_img_bgr, (x1, y1), (x2, y2), color, 2)

            # Retrieve depth value for the object
            depth_text = f"Depth: {depth_values.get(cls, 'N/A'):.2f}m"
            cv2.putText(left_img_bgr, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display images
        cv2.imshow("ZED Left Image", left_img_bgr)
        cv2.imshow("ZED Right Image", right_img_bgr)
        cv2.imshow("ZED Depth Map", depth_map_colored)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


# Main function to run the live feed
if __name__ == "__main__":
    zed = init_camera()
    display_live_feed(zed)
    zed.close()

