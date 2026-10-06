import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

CALIBRATION_FILE = "camera_calibration.npz"

# Length of displayed coordinate axes,
# measured in number of board squares.
AXIS_LENGTH_IN_SQUARES = 3


# Marker size relative to chessboard square size.
#
# Example:
#   square = 25 mm
#   marker = 17.5 mm
#
MARKER_SIZE_RATIO = 0.70

# ArUco dictionary used by the board.
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

calibration = np.load(CALIBRATION_FILE)

camera_matrix = calibration["camera_matrix"]

# Support either naming convention
if "distortion_coefficients" in calibration:
    dist_coeffs = calibration["distortion_coefficients"]

elif "dist_coeffs" in calibration:
    dist_coeffs = calibration["dist_coeffs"]

else:
    raise KeyError(
        "Could not find distortion coefficients in calibration file."
    )


# ============================================================
# LOAD BOARD INFORMATION
# ============================================================

# These were saved when calibrating with the normal checkerboard.
#
# Old checkerboard:
#
#     cols = number of INTERNAL corners across
#     rows = number of INTERNAL corners down
#
# ChArUcoBoard instead takes the number of SQUARES.
#
# Therefore:
#
#     squares_x = cols + 1
#     squares_y = rows + 1

if "chessboard_cols" in calibration:
    cols = int(calibration["chessboard_cols"])

else:
    # Change this if it was not saved in your calibration file
    cols = 9


if "chessboard_rows" in calibration:
    rows = int(calibration["chessboard_rows"])

else:
    # Change this if it was not saved in your calibration file
    rows = 6


if "square_size" in calibration:
    square_length = float(calibration["square_size"])

else:
    # IMPORTANT:
    # Change this to the REAL physical side length of one square.
    #
    # Example:
    # 0.025 means 25 mm if using meters.
    square_length = 20 #0.025


# ChArUco uses number of squares rather than number
# of internal chessboard corners.
squares_x = cols + 1
squares_y = rows + 1

marker_length = square_length * MARKER_SIZE_RATIO


# ============================================================
# CREATE CHARUCO BOARD
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    dictionary
)


# ============================================================
# CREATE CHARUCO DETECTOR
# ============================================================

charuco_params = cv2.aruco.CharucoParameters()

# Supplying camera calibration helps the ChArUco detector
# interpolate the chessboard corners more accurately.
charuco_params.cameraMatrix = camera_matrix
charuco_params.distCoeffs = dist_coeffs

# Try to recover markers that may initially have been rejected.
charuco_params.tryRefineMarkers = True


detector_params = cv2.aruco.DetectorParameters()


charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charuco_params,
    detector_params
)

# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Could not open camera.")
    exit()

# ============================================================
# SET CAMERA TO CALIBRATED RESOLUTION
# ============================================================

# Try several possible names depending on how the original
# calibration file was saved.

if (
    "image_width" in calibration
    and
    "image_height" in calibration
):

    calibrated_width = int(
        calibration["image_width"]
    )

    calibrated_height = int(
        calibration["image_height"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibrated_width
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibrated_height
    )

elif "resolution" in calibration:

    resolution = calibration["resolution"]

    calibrated_width = int(resolution[0])
    calibrated_height = int(resolution[1])

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibrated_width
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibrated_height
    )

# ============================================================
# AXIS LENGTH
# ============================================================
axis_length = (
    square_length
    * AXIS_LENGTH_IN_SQUARES
)

# ============================================================
# DISPLAY BOARD INFORMATION
# ============================================================

print()
print("========================================")
print("ChArUco board configuration")
print("========================================")

print(
    f"Squares: {squares_x} x {squares_y}"
)

print(
    f"Square length: {square_length}"
)

print(
    f"Marker length: {marker_length}"
)

print(
    f"Possible ChArUco corners: "
    f"{(squares_x - 1) * (squares_y - 1)}"
)

print("Press q to quit.")
print()

# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    # ========================================================
    # DETECT CHARUCO BOARD
    # ========================================================
    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids

    ) = charuco_detector.detectBoard(frame)


# --------------------------------------------------------
# Normalize ChArUco output format for OpenCV
# --------------------------------------------------------

    if charuco_corners is not None and charuco_ids is not None:

        charuco_corners = np.asarray(
            charuco_corners,
            dtype=np.float32
        ).reshape(-1, 1, 2)

        charuco_ids = np.asarray(
            charuco_ids,
            dtype=np.int32
        ).reshape(-1, 1)

    if charuco_corners is not None:
        print(
            "corners:",
            charuco_corners.shape,
            charuco_corners.dtype,
            "| ids:",
            charuco_ids.shape if charuco_ids is not None else None,
            charuco_ids.dtype if charuco_ids is not None else None
        )


    # ========================================================
    # DRAW DETECTED ARUCO MARKERS
    # ========================================================
    if (
        marker_ids is not None
        and len(marker_ids) > 0
    ):

        cv2.aruco.drawDetectedMarkers(
            frame,
            marker_corners,
            marker_ids
        )

    # ========================================================
    # DRAW DETECTED CHARUCO CORNERS
    # ========================================================
    if (
        charuco_ids is not None
        and len(charuco_ids) > 0
    ):

        cv2.aruco.drawDetectedCornersCharuco(
            frame,
            charuco_corners,
            charuco_ids
        )

    # ========================================================
    # POSE ESTIMATION
    # ========================================================
    valid_pose = False

    number_of_corners = 0

    if charuco_ids is not None:

        number_of_corners = len(charuco_ids)

        # ----------------------------------------------------
        # Need at least 4 known 2D <-> 3D correspondences
        # ----------------------------------------------------
        if number_of_corners >= 4:
            # ------------------------------------------------
            # Make sure the corners are not all on one line.
            #
            # Four points mathematically exist, but if all four
            # are collinear, solvePnP cannot determine the pose.
            # ------------------------------------------------
            collinear = (
                board.checkCharucoCornersCollinear(
                    charuco_ids
                )
            )

            if not collinear:
                # ============================================
                # GET MATCHING 3D AND 2D POINTS
                # ============================================

                #
                # This is the important ChArUco step.
                #
                # charuco_ids tell OpenCV WHICH board
                # corners were detected.
                #
                # Therefore we only create correspondences
                # for the visible corners.
                #
                object_points, image_points = (
                    board.matchImagePoints(
                        charuco_corners,
                        charuco_ids
                    )
                )

                # ============================================
                # SOLVE PNP
                # ============================================
                success_pnp, rvec, tvec = (
                    cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE# cv2.SOLVEPNP_ITERATIVE
                    )
                )


                if success_pnp:
                    valid_pose = True

                    # ========================================
                    # CONVERT ROTATION VECTOR
                    # TO ROTATION MATRIX
                    # ========================================
                    R, _ = cv2.Rodrigues(rvec)

                    # ========================================
                    # CAMERA POSITION IN BOARD COORDINATES
                    #
                    # solvePnP gives:
                    #
                    #     Pc = R Pw + t
                    #
                    # Camera center C satisfies:
                    #
                    #     0 = R C + t
                    #
                    # Therefore:
                    #
                    #     C = -R^T t
                    # ========================================
                    camera_position = (
                        -R.T @ tvec
                    )

                    x = camera_position[0, 0]
                    y = camera_position[1, 0]
                    z = camera_position[2, 0]


                    # ------------------------------------------------
                    # Rcamera=R^(-1)=R.T
                    # Rcam=[xc, yc, zc], xc=>direction of camera's right axis, yc=>down axis, zc=>forward axis
                    # ------------------------------------------------
        
                    Rcam = R.T
        
                    roll=np.degrees(np.arctan2(Rcam[2,1], Rcam[2,2]))
                    pitch=np.degrees(np.arcsin((-1)*Rcam[2,0]))
                    yaw=np.degrees(np.arctan2(Rcam[1,0], Rcam[0,0]))
        
                    camroll=yaw
                    campitch=roll
                    camyaw=pitch

                    # ========================================
                    # DRAW BOARD COORDINATE AXES
                    # ========================================
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        axis_length,
                        3
                    )

                    # ========================================
                    # DISPLAY CAMERA POSITION
                    # ========================================
                    cv2.putText(
                        frame,
                        f"X: {x:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Y: {y:.3f}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Z: {z:.3f}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 0),
                        2
                    )

                    # ------------------------------------------------
                    # Display orientation
                    # ------------------------------------------------
        
                    cv2.putText(
                        frame,
                        f"Roll: {camroll:.1f}",
                        (1120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
        
                    cv2.putText(
                        frame,
                        f"Pitch: {campitch:.1f}",
                        (1120, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
        
                    cv2.putText(
                        frame,
                        f"Yaw: {camyaw:.1f}",
                        (1120, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    # ========================================
                    # DISPLAY NUMBER OF VISIBLE CORNERS
                    # ========================================
                    cv2.putText(
                        frame,
                        f"ChArUco corners: "
                        f"{number_of_corners}",
                        (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Pose estimated",
                        (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # ========================================
                    # PRINT POSITION TO TERMINAL
                    # ========================================
                    print(
                        f"\rCorners: "
                        f"{number_of_corners:2d} | "
                        f"X: {x:8.3f} | "
                        f"Y: {y:8.3f} | "
                        f"Z: {z:8.3f}",
                        end=""
                    )

    # ========================================================
    # NO VALID POSE
    # ========================================================
    if not valid_pose:
        cv2.putText(
            frame,
            f"ChArUco corners: "
            f"{number_of_corners}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Pose unavailable",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ========================================================
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "ChArUco Camera Pose",
        frame
    )
    # ========================================================
    # QUIT WITH Q
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEAN UP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera closed.")