import cv2
import numpy as np
import math

# ============================================================
# SETTINGS
# ============================================================

CALIBRATION_FILE = "camera_calibration.npz"
CAMERA_INDEX = 1

# Length of the coordinate axes drawn on the board.
# This uses the SAME unit as SQUARE_SIZE.
# For example, if square_size is in mm, this is also mm.
AXIS_LENGTH_IN_SQUARES = 3


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

data = np.load(CALIBRATION_FILE)

camera_matrix = data["camera_matrix"]
dist_coeffs = data["distortion_coefficients"]

image_width = int(data["image_width"])
image_height = int(data["image_height"])

cols = int(data["chessboard_cols"])
rows = int(data["chessboard_rows"])

square_size = float(data["square_size"])

chessboard_size = (cols, rows)

print("Loaded calibration:")
print()
print("Camera matrix:")
print(camera_matrix)

print()
print("Distortion coefficients:")
print(dist_coeffs)

print()
print(f"Resolution: {image_width} x {image_height}")
print(f"Chessboard internal corners: {cols} x {rows}")
print(f"Square size: {square_size}")


# ============================================================
# CREATE KNOWN 3D CHESSBOARD POINTS
# ============================================================

# These are the coordinates of every internal chessboard corner
# in the board coordinate system.
#
# The first detected internal corner is:
#
#     (0, 0, 0)
#
# X increases across the board.
# Y increases down the board.
# Z is perpendicular to the board.

object_points = np.zeros((rows * cols, 3), np.float32)

object_points[:, :2] = (
    np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
)

object_points *= square_size


# ============================================================
# SUB-PIXEL CORNER REFINEMENT
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# AXES FOR VISUALIZATION
# ============================================================

axis_length = square_size * AXIS_LENGTH_IN_SQUARES

axis_points = np.float32([
    [0, 0, 0],              # origin
    [axis_length, 0, 0],    # X axis
    [0, axis_length, 0],    # Y axis
    [0, 0, -axis_length]    # Z axis
])


# ============================================================
# 2D vector to store angle values
# ============================================================

angles = np.zeros((3,3))

# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print()
print(f"Camera opened at {actual_width} x {actual_height}")

if actual_width != image_width or actual_height != image_height:
    print()
    print("WARNING:")
    print("Camera resolution does not match calibration resolution.")
    print("Pose estimates may be incorrect.")

print()
print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Detect chessboard
    # --------------------------------------------------------

    # found, corners = cv2.findChessboardCorners(
    #     gray,
    #     chessboard_size,
    #     flags=(
    #         cv2.CALIB_CB_ADAPTIVE_THRESH
    #         + cv2.CALIB_CB_NORMALIZE_IMAGE
    #     )
    # )

    # using normal find corners, the board may not be detectable if two external corner pts of the same corner block isn't visible

    found, corners = cv2.findChessboardCornersSB(
            gray,
            chessboard_size,
            flags=(
                cv2.CALIB_CB_LARGER
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_ACCURACY
            )
        )

    if found:

        # ----------------------------------------------------
        # Refine corner locations
        # ----------------------------------------------------

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(
            frame,
            chessboard_size,
            refined_corners,
            found
        )

        # ----------------------------------------------------
        # Solve PnP
        #
        # This returns the transform:
        #
        # world/board -> camera
        #
        # Pc = R * Pw + t
        # ----------------------------------------------------

        success_pnp, rvec, tvec = cv2.solvePnP(
            object_points,
            refined_corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success_pnp:

            # ------------------------------------------------
            # Convert rotation vector to rotation matrix
            # ------------------------------------------------

            R, _ = cv2.Rodrigues(rvec)

            # ------------------------------------------------
            # CAMERA POSITION IN BOARD COORDINATES
            #
            # Pc = R * Pw + t
            #
            # Camera center:
            #
            # C = -R^T * t
            # ------------------------------------------------

            camera_position = -R.T @ tvec

            x = camera_position[0, 0]
            y = camera_position[1, 0]
            z = camera_position[2, 0]

            # ------------------------------------------------
            # Draw coordinate axes
            # ------------------------------------------------

            projected_axes, _ = cv2.projectPoints(
                axis_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )

            pts = projected_axes.reshape(-1, 2).astype(int)

            origin = tuple(pts[0])

            # X
            cv2.line(
                frame,
                origin,
                tuple(pts[1]),
                (0, 0, 255),
                3
            )

            # Y
            cv2.line(
                frame,
                origin,
                tuple(pts[2]),
                (0, 255, 0),
                3
            )

            # Z
            cv2.line(
                frame,
                origin,
                tuple(pts[3]),
                (255, 0, 0),
                3
            )

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

            angles = np.arccos(Rcam)

            # key1 = cv2.waitKey(1) & 0xFF
            # if key1==ord("f"):
                # print(f"\nx: {Rcam[:,0]}")
                # print(f"y: {Rcam[:,1]}")
                # print(f"z: {Rcam[:,2]}\n")

                # print(np.matrix(Rcam))

                # print(f"\n{np.degrees(angles)}\n")
                # print(f"roll:{camroll}    pitch:{campitch}    yaw:{camyaw}\n")


            # ------------------------------------------------
            # Display position
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"X: {x:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Y: {y:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Z: {z:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Chessboard detected",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print(
                f"\rCamera position: "
                f"X={x:8.2f}, "
                f"Y={y:8.2f}, "
                f"Z={z:8.2f}"
                f"      roll:{camroll:.0f}, "
                f"pitch:{campitch:.0f}, "
                f"yaw:{camyaw:.0f}",
                end=""
            )

    else:

        cv2.putText(
            frame,
            "Chessboard not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # Show live frame
    # --------------------------------------------------------

    cv2.imshow("Live Camera Pose", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nFinished.")