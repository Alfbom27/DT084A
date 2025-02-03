import numpy as np
import cv2

# CV2 ORB implementation
def cv2_ORB(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB.create()
    # Feature detection
    keypoints = orb.detect(image_gray, None)
    # Keypoints descriptors
    keypoints, descriptors = orb.compute(image_gray, keypoints)
    # Draw keypoints on the image
    img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    return img


def harris_corner_detection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Smoothing the image, sometimes not required.
    # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Normalization
    img_gray = np.float32(img_gray)
    img_gray = img_gray / 255.0
    # Vertical and horizontal Sobel operators
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    # X and Y - derivatives by convolving the sobel-operators with the image
    gradient_x = cv2.filter2D(img_gray, -1, sobel_x)
    gradient_y = cv2.filter2D(img_gray, -1, sobel_y)

    # Components in Harris matrix. Same dimension as original image
    ixx = gradient_x ** 2
    iyy = gradient_y ** 2
    ixy = gradient_x * gradient_y

    # Smoothing to reduce noise and prevent poor detections.
    ixx = cv2.GaussianBlur(ixx, (3, 3), 0)
    iyy = cv2.GaussianBlur(iyy, (3, 3), 0)
    ixy = cv2.GaussianBlur(ixy, (3, 3), 0)

    # Harris matrix
    # harris_matrix = np.array([[ixx, ixy], [ixy, iyy]])

    # Matrix determinant and trace
    det_harris = ixx*iyy - ixy**2
    trace_harris = ixx + iyy

    # Corner response function. Image-sized grid with scores of pixel belonging to a corner or not
    r_value = det_harris - 0.01*trace_harris**2
    # Makes values in the r-value matrix larger
    corner_response = cv2.dilate(r_value, None)
    # Corner response values above a threshold is marked red in the original image
    img[corner_response > 0.01 * corner_response.max()] = [0, 0, 255]
    return img

# Image file path
image = cv2.imread("./cube.png")
dims = image.shape

image_corner = harris_corner_detection(image.copy())
image_orb = cv2_ORB(image.copy())

# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Harris Corner Detector", cv2.WINDOW_NORMAL)
cv2.namedWindow("ORB", cv2.WINDOW_NORMAL)

# cv2.resizeWindow("Original", dims[0], dims[1])
cv2.resizeWindow("Harris Corner Detector", dims[0], dims[1])
cv2.resizeWindow("ORB", dims[0], dims[1])

# cv2.imshow("Original", image)
cv2.imshow("Harris Corner Detector", image_corner)
cv2.imshow("ORB", image_orb)

cv2.waitKey(0)
cv2.destroyAllWindows()