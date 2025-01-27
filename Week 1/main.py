import cv2
import sys

import numpy as np


def do_nothing(x):
    pass

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    return cap


# Convolves the image processing kernels with the current frame and adjusts brightness.
def process_frame(frame, brightness, sharpening, edge_detection, box_blur):
    frame = cv2.filter2D(frame, -1, sharpening)
    frame = cv2.filter2D(frame, -1, edge_detection)
    frame = cv2.filter2D(frame, -1, box_blur)
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness-50)
    return frame


def main():
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    print("Camera feed started. Press 'q' to quit.")
    cv2.namedWindow('Original Frame')
    # Creating track-bars for image processing
    cv2.createTrackbar('Brightness', 'Original Frame', 50, 100, do_nothing)
    cv2.createTrackbar('Sharpening', 'Original Frame', 0, 1, do_nothing)
    cv2.createTrackbar('Edge detection', 'Original Frame', 0, 1, do_nothing)
    cv2.createTrackbar('Box blur', 'Original Frame', 0, 1, do_nothing)
    cv2.resizeWindow('Original Frame', 1200, 900)

    # Defining image processing kernels
    sharpening_kernel = np.array([[ 0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    edge_detection_kernel = np.array([[ 0, -1, 0],
                                [-1, 4, -1],
                                [0, -1, 0]])

    box_blur_kernel = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])/9

    identity_kernel = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]])

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Get current value of the track-bar
        brightness = cv2.getTrackbarPos('Brightness', 'Original Frame')
        apply_sharpening = cv2.getTrackbarPos('Sharpening', 'Original Frame')
        apply_edge_detection = cv2.getTrackbarPos('Edge detection', 'Original Frame')
        apply_blur = cv2.getTrackbarPos('Box blur', 'Original Frame')

        # Apply the kernels depending on track-bar values
        sharpening = sharpening_kernel if apply_sharpening else identity_kernel
        edge_detection = edge_detection_kernel if apply_edge_detection else identity_kernel
        box_blur = box_blur_kernel if apply_blur else identity_kernel

        # Process the current frame
        output_frame = process_frame(frame, brightness, sharpening, edge_detection, box_blur)

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Display the processed frame
        cv2.imshow('Processed Frame', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()