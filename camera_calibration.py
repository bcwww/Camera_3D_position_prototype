import cv2
import numpy as np
import os

# ============================================================
# SETTINGS YOU SHOULD CHANGE
# ============================================================

# Number of INTERNAL chessboard corners, not number of squares.
# Example:
#   10 squares x 7 squares -> 9 x 6 internal corners
CHESSBOARD_SIZE = (10, 7)

# Physical size of ONE chessboard square.
# Use whatever unit you want your future position estimates in.
# Example: 25.0 mm
SQUARE_SIZE = 20.0

# Camera index:
# 0 is usually the first/default webcam.
CAMERA_INDEX = 0

# Resolution used for calibration.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Number of good calibration views to collect.
TARGET_IMAGES = 20

OUTPUT_FILE = "camera_calibration.npz"


# ============================================================
# PREPARE KNOWN 3D CHESSBOARD POINTS
# ============================================================

cols, rows = CHESSBOARD_SIZE

# Example for a 9 x 6 board:
#
# (0,0,0), (25,0,0), (50,0,0), ...
# (0,25,0), (25,25,0), ...
#
# All Z coordinates are 0 because the chessboard is planar.

object_points_template = np.zeros((rows * cols, 3), np.float32)

object_points_template[:, :2] = (
    np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE


# Stores the known 3D locations for every successful image.
object_points = []

# Stores detected 2D pixel coordinates for every successful image.
image_points = []


# ============================================================
# CORNER REFINEMENT SETTINGS
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera opened.")
print(f"Requested resolution: {FRAME_WIDTH} x {FRAME_HEIGHT}")
print(f"Actual resolution:    {actual_width} x {actual_height}")
print()
print("Instructions:")
print("  Move the chessboard to different positions and angles.")
print("  Press SPACE when the chessboard is detected to save that view.")
print("  Press Q to finish early.")
print()


# ============================================================
# COLLECT CALIBRATION IMAGES
# ============================================================

while len(image_points) < TARGET_IMAGES:

    success, frame = cap.read()

    if not success:
        print("Failed to read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    )

    display = frame.copy()

    refined_corners = None

    if found:

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(
            display,
            CHESSBOARD_SIZE,
            refined_corners,
            found
        )

        status = "Chessboard detected - press SPACE to capture"

    else:
        status = "Chessboard not detected"

    cv2.putText(
        display,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if found else (0, 0, 255),
        2
    )

    cv2.putText(
        display,
        f"Images: {len(image_points)}/{TARGET_IMAGES}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32 and found:

        object_points.append(object_points_template.copy())
        image_points.append(refined_corners.copy())

        print(
            f"Captured calibration image "
            f"{len(image_points)}/{TARGET_IMAGES}"
        )

    # Q
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


# ============================================================
# CHECK THAT ENOUGH IMAGES WERE COLLECTED
# ============================================================

if len(image_points) < 10:
    raise RuntimeError(
        f"Only {len(image_points)} usable images were collected. "
        "Try to collect at least 10, preferably around 15-25."
    )


# ============================================================
# CAMERA CALIBRATION
# ============================================================

image_size = (actual_width, actual_height)

rms_error, camera_matrix, distortion_coefficients, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)


# ============================================================
# CALCULATE MEAN REPROJECTION ERROR
# ============================================================

total_error = 0.0

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coefficients
    )

    detected = image_points[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    error = np.linalg.norm(
        detected - projected
    ) / len(projected)

    # error = cv2.norm(
    #     image_points[i],
    #     projected_points,
    #     cv2.NORM_L2
    # ) / len(projected_points)

    total_error += error

mean_reprojection_error = total_error / len(object_points)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================================")
print("CALIBRATION COMPLETE")
print("==============================================")

print("\nImage resolution:")
print(image_size)

print("\nCamera matrix K:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion_coefficients)

print("\nRMS calibration error:")
print(rms_error)

print("\nMean reprojection error:")
print(mean_reprojection_error)

print("\nIntrinsic parameters:")

print(f"fx = {camera_matrix[0, 0]}")
print(f"fy = {camera_matrix[1, 1]}")

print(f"cx = {camera_matrix[0, 2]}")
print(f"cy = {camera_matrix[1, 2]}")

print("\nDistortion parameters:")

dist = distortion_coefficients.ravel()

names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

for i, value in enumerate(dist):
    name = names[i] if i < len(names) else f"d{i}"
    print(f"{name} = {value}")


# ============================================================
# SAVE PARAMETERS
# ============================================================

np.savez(
    OUTPUT_FILE,

    camera_matrix=camera_matrix,

    distortion_coefficients=distortion_coefficients,

    image_width=actual_width,
    image_height=actual_height,

    chessboard_cols=cols,
    chessboard_rows=rows,

    square_size=SQUARE_SIZE,

    rms_error=rms_error,
    mean_reprojection_error=mean_reprojection_error
)

print(f"\nCalibration saved to: {OUTPUT_FILE}")