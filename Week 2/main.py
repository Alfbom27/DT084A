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



# Applies high-pass filter
def apply_high_pass(frame, mask_radius):
    mask = np.ones_like(frame)
    y = mask.shape[0] // 2
    x = mask.shape[1] // 2
    # Creates a black circle in the middle of the mask
    cv2.circle(mask, (x, y), mask_radius, (0,0,0), -1)
    # Fourier transform across each color channel and shifts low frequencies to the center
    ft = np.fft.fft2(frame, axes=(0,1))
    ft = np.fft.fftshift(ft)

    # Applies mask and transfers back to spatial domain
    filtered_ft = ft * mask
    filtered_ft = np.fft.ifftshift(filtered_ft)
    filtered = np.fft.ifft2(filtered_ft, axes=(0,1))

    # Takes the real part from the inverse-FT and scales the image pixel values between 0 and 255
    filtered = np.abs(2*filtered).clip(0,255).astype(np.uint8)
    return filtered

# Process the current frame
def process_frame(frame, radius):
    frame = apply_high_pass(frame, radius)
    return frame


def main():
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    print("Camera feed started. Press 'q' to quit.")
    cv2.namedWindow('Original Frame')
    # Creating track-bars for image processing
    cv2.createTrackbar('Radius', 'Original Frame', 32, 64, do_nothing)
    cv2.resizeWindow('Original Frame', 1200, 900)


    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        radius = cv2.getTrackbarPos('Radius', 'Original Frame')

        # Process the current frame
        output_frame = process_frame(frame, radius)

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