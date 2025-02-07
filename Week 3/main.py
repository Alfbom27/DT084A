import cv2
import sys
import numpy as np
import time

def do_nothing(x):
    pass

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    return cap


# Applies high-pass filter
def apply_high_pass(frame):
    mask_radius = cv2.getTrackbarPos('Radius', 'Original Frame') * 2
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


def feature_detection(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB.create()
    # Feature detection
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
    # Draw keypoints in the image
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
    return frame


def feature_matching(frame, saved_frame, settings):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saved_gray = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2GRAY)
    # ORB feature detection and description
    orb = cv2.ORB.create()
    keypoints1, descriptors1 = orb.detectAndCompute(frame_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(saved_gray, None)

    # Brute force matching. Uses crossCheck, to ensure bidirectional matching of keypoints
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feature_matches = bf_matcher.match(descriptors1, descriptors2)

    # Sort features, best to worst
    feature_matches = sorted(feature_matches, key=lambda x: x.distance)

    # Stores the matched keypoint positions of the first 30 matches of both the source and destination image in two arrays
    source_points = np.float32([keypoints1[match.queryIdx].pt for match in feature_matches[:30]]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints2[match.trainIdx].pt for match in feature_matches[:30]]).reshape(-1, 1, 2)

    # Concatenate image horizontally
    combined_image = cv2.hconcat([frame, saved_frame])

    good_matches = feature_matches[:30]
    # Use RANSAC, filter outliers and only keep inliers.
    if settings["ransac"]:
        if len(source_points) >= 4 and len(destination_points) >= 4:
            if settings["transformation_type"] == "homography":
                h_matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC)
                good_matches = [feature_matches[i] for i in range(len(mask)) if mask[i] == 1]
            elif settings["transformation_type"] == "affine":
                a_matrix, mask = cv2.estimateAffine2D(source_points, destination_points, method=cv2.RANSAC)
                good_matches = [feature_matches[i] for i in range(len(mask)) if mask[i] == 1]

    cv2.drawMatches(frame, keypoints1, saved_frame, keypoints2, good_matches, combined_image, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return combined_image




def switch_program(key):
    if key == ord('0'):
        return 0
    elif key == ord('1'):
        return 1
    elif key == ord('2'):
        return 2
    elif key == ord('3'):
        return 3


def display_text(frame, program, settings):
    menu_text = "[1] High pass filter\n[2] Feature detection\n[3] Feature matching\n[Q] Exit"
    lines = menu_text.split("\n")
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    y_position = 15
    for line in lines:
        cv2.putText(frame, line, (10, y_position), font, font_scale, color, thickness)
        y_position += 15

    if program == 3:
        ransac_text = "[R] RANSAC: "
        if settings["ransac"]:
            ransac_text += "ON"
        else:
            ransac_text += "OFF"
        cv2.putText(frame, ransac_text, (490,15), font, font_scale, color, thickness)
        transformation_text = "[S] "
        if settings["transformation_type"] == "homography":
            transformation_text += "Homography"
        elif settings["transformation_type"] == "affine":
            transformation_text += "Affine"
        cv2.putText(frame, transformation_text, (490, 30), font, font_scale, color, thickness)





def init_program(program):
    cv2.namedWindow('Original Frame')
    if program == 1:
        cv2.createTrackbar('Radius', 'Original Frame', 10, 20, do_nothing)
        cv2.setTrackbarPos('Radius', 'Original Frame', 10)


def main():
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    program = 0
    init_program(program)
    print("Camera feed started. Press 'q' to quit.")

    saved_frame = None
    settings = {
        "ransac": False,
        "display_fps": False,
        "transformation_type": "homography",
    }
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        if program == 0:
            display_text(frame, program, settings)
            cv2.imshow('Original Frame', frame)
        elif program == 1:
            processed_frame = apply_high_pass(frame)
            display_text(frame, program, settings)
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Processed Frame', processed_frame)
        elif program == 2:
            processed_frame = feature_detection(frame)
            display_text(processed_frame, program, settings)
            cv2.imshow('Original Frame', processed_frame)
        elif program == 3:
            if saved_frame is None:
                saved_frame = frame.copy()
            processed_frame = feature_matching(frame, saved_frame, settings)
            display_text(processed_frame, program, settings)
            cv2.imshow('Original Frame', processed_frame)


        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            break
        elif key_press == ord('c'):
            if program == 3:
                saved_frame = frame.copy()
        elif key_press == ord('r'):
            if program == 3:
                settings["ransac"] = not settings["ransac"]
        elif key_press == ord('s'):
            if program == 3:
                if settings["transformation_type"] == "homography":
                    settings["transformation_type"] = "affine"
                elif settings["transformation_type"] == "affine":
                    settings["transformation_type"] = "homography"

        else:
            new_program = switch_program(key_press)
            if new_program is not None:
                program = new_program
                cv2.destroyAllWindows()
                init_program(program)
                print(program)







    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()