import numpy as np
import cv2

# Interpolates the value of the point (x,y) by considering the values of its four surrounding neighbours.
def bilinear_interpolation(image, x, y):
    h, w = image.shape[:2]

    # Corner x-y points
    x1, x2 = int(np.floor(x)), int(np.ceil(x))
    y1, y2 = int(np.floor(y)), int(np.ceil(y))

    # Boundary check
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)

    # Assigning corner values
    corner1 = image[y1, x1]
    corner2 = image[y2, x1]
    corner3 = image[y1, x2]
    corner4 = image[y2, x2]

    # x and y-differences
    dx = x - x1
    dy = y - y1

    # Interpolation
    top = (1 - dx) * corner1 + dx * corner3
    bottom = (1 - dx) * corner2 + dx * corner4
    pixel = (1 - dy) * top + dy * bottom
    return pixel


def backward_warping(H_inv, stitched_image, source_image):
    # Backward warping / Inverse warping
    for y in range(stitched_image.shape[0]):
        for x in range(0, stitched_image.shape[1]):
            # Point in reference image
            pt_p = np.array([x, y, 1])
            # Compute its location in the target image using the inverse homography matrix
            pt = np.dot(H_inv, pt_p)
            # To cartesian coordinates
            pt = pt / pt[2]

            x_1, y_1 = pt[0], pt[1]
            if 0 <= x_1 < source_image.shape[1] and 0 <= y_1 < source_image.shape[0]:
                # Interpolate pixel value of the fractional coordinate
                pixel_value = bilinear_interpolation(source_image, x_1, y_1)
                stitched_image[y][x] = pixel_value

    return stitched_image


def forward_warping(H, stitched_image, source_image):
    # Forward warping
    for y in range(source_image.shape[0]):
        for x in range(0, source_image.shape[1]):
            # Point in the source/target image
            pt_p = np.array([x, y, 1])
            # Compute its position in the reference image using the homography
            pt = np.dot(H, pt_p)
            # To cartesian coordinates
            pt = pt / pt[2]

            # Rounding fractional coordinates
            x_1, y_1 = int(np.round(pt[0])), int(np.round(pt[1]))

            # Assign pixel value if located within boundaries
            if 0 <= x_1 < stitched_image.shape[1] and 0 <= y_1 < stitched_image.shape[0]:
                pixel_value = source_image[y][x]
                stitched_image[y_1][x_1] = pixel_value

    return stitched_image


destination_image = cv2.imread('viewpoint_1.png')
source_image = cv2.imread('viewpoint_2.png')

dst_gray = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT.create()

# Feature point detection using SIFT
feature_points1, descriptors1 = sift.detectAndCompute(source_gray, None)
feature_points2, descriptors2 = sift.detectAndCompute(dst_gray, None)


# Brute force matching
bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
feature_matches = bf_matcher.match(descriptors1, descriptors2)


feature_matches = sorted(feature_matches, key=lambda x: x.distance)

# Stores the matched keypoint positions of both the source and destination image in two arrays
source_points = np.float32([feature_points1[match.queryIdx].pt for match in feature_matches]).reshape(-1, 1, 2)
destination_points = np.float32([feature_points2[match.trainIdx].pt for match in feature_matches]).reshape(-1, 1, 2)

# RANSAC to filter outliers
h_matrix, mask = cv2.findHomography(source_points, destination_points, method=cv2.RANSAC)
good_matches = [feature_matches[i] for i in range(len(mask)) if mask[i] == 1]


points1 = []
points2 = []
# Extract the matched feature points
for match in good_matches:
    pt1 = feature_points1[match.queryIdx].pt
    pt2 = feature_points2[match.trainIdx].pt
    points1.append(pt1)
    points2.append(pt2)

# Construct the A-matrix with the matched points
A = []
for pt1, pt2 in zip(points1, points2):
    x_1, y_1 = np.float32(pt1)
    x_prime, y_prime = np.float32(pt2)
    A.append([x_1, y_1, 1, 0, 0, 0, -x_1*x_prime, -y_1*x_prime, -x_prime])
    A.append([0, 0, 0, x_1, y_1, 1, -x_1*y_prime, -y_1*y_prime, -y_prime])


# Singular Value Decomposition
U, D, V_t = np.linalg.svd(A)

# Transpose back
V = V_t.T

# Take the last column of V, the least square solution to Ah=0 - The resulting homography
h = V[:,-1]
H = h.reshape(3,3)

print(f'Homography: {H}')


height1, width1 = destination_image.shape[:2]
height2, width2 = source_image.shape[:2]


stitched_image = np.full((height1, width1 * 2, 3), 0, dtype=np.uint8)

# Inverse homography
H_inv = np.linalg.inv(H)

# Forward or backward warping
stitched_image = backward_warping(H_inv, stitched_image, source_image)
# stitched_image = forward_warping(H, stitched_image, source_image)

stitched_image[:height1, :width1] = destination_image


cv2.namedWindow("Stitched Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stitched Image", stitched_image.shape[1], stitched_image.shape[0])
cv2.imshow('Stitched Image', stitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()