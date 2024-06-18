import sys
import cv2
import numpy as np
import time
import imutils
import triangulation as tri
import HSV_filter as hsv

# Load pre-trained SSD model and class labels/Users/tanmay/practice/stereovision/MobilNet_SSD_opencv/MobileNetSSD_deploy.prototxt
proto_file = '/Users/tanmay/practice/stereovision/MobilNet_SSD_opencv/MobileNetSSD_deploy.prototxt'
model_file = '/Users/tanmay/practice/stereovision/MobilNet_SSD_opencv/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
classNames = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

cap_right = cv2.VideoCapture(0)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap_left = cv2.VideoCapture(1)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

B = 15  # Baseline distance between the two cameras
f = 24  # Focal length of the camera
alpha = 55  # Field of view in degrees

input_shape = (300, 300)
mean = (127.5, 127.5, 127.5)
scale = 0.007843

# Initial values
count = -1

while True:
    count += 1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    # If cannot catch any frame, break
    if not ret_right or not ret_left:
        break

    else:
        # APPLYING HSV-FILTER:
        mask_right = hsv.add_HSV_filter(frame_right, 1)
        mask_left = hsv.add_HSV_filter(frame_left, 0)

        # Result-frames after applying HSV-filter mask
        res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
        res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

        # Prepare the frames for SSD object detection
        blob_right = cv2.dnn.blobFromImage(frame_right, scalefactor=scale, size=input_shape, mean=mean, swapRB=True)
        blob_left = cv2.dnn.blobFromImage(frame_left, scalefactor=scale, size=input_shape, mean=mean, swapRB=True)

        # Set the blob as input to the network
        net.setInput(blob_right)
        detections_right = net.forward()

        net.setInput(blob_left)
        detections_left = net.forward()

        # Function to find the most confident detection
        def find_best_detection(detections, confidence_threshold=0.7):
            best_confidence = 0
            best_box = None
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold and confidence > best_confidence:
                    best_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([frame_right.shape[1], frame_right.shape[0], frame_right.shape[1], frame_right.shape[0]])
                    best_box = box.astype("int")
            return best_box

        box_right = find_best_detection(detections_right)
        box_left = find_best_detection(detections_left)

        if box_right is None or box_left is None:
            cv2.putText(frame_right, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_left, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Draw the bounding boxes
            cv2.rectangle(frame_right, (box_right[0], box_right[1]), (box_right[2], box_right[3]), (0, 255, 0), 2)
            cv2.rectangle(frame_left, (box_left[0], box_left[1]), (box_left[2], box_left[3]), (0, 255, 0), 2)

            # Calculate the center points of the detected object
            center_right = ((box_right[0] + box_right[2]) // 2, (box_right[1] + box_right[3]) // 2)
            center_left = ((box_left[0] + box_left[2]) // 2, (box_left[1] + box_left[3]) // 2)

            # Calculate depth of object
            depth = tri.find_depth(center_right, center_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(frame_left, "TRACKING", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(frame_right, "Distance: " + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            cv2.putText(frame_left, "Distance: " + str(round(depth, 3)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
            print("Depth: ", depth)

        # Show the frames
        cv2.imshow("frame right", frame_right)
        cv2.imshow("frame left", frame_left)
        cv2.imshow("mask right", mask_right)
        cv2.imshow("mask left", mask_left)

        # Hit "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
