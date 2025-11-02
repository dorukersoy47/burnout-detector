"""
Simple eye-tracking using MediaPipe FaceMesh (iris landmarks).

This script captures from the webcam, detects iris landmarks and computes
an approximate gaze point. Every 5 seconds it prints whether the gaze is
"in screen" or "outside of screen".

Usage:
    python3 motion/eye.py

Press 'q' to quit.
"""
import time
import argparse
import sys

import cv2
import mediapipe as mp


def classify_gaze(norm_x, norm_y, margin=0.05):
    """Classify gaze position relative to the screen using normalized coords.

    Args:
        norm_x: normalized x in [0,1] (0=left,1=right)
        norm_y: normalized y in [0,1] (0=top,1=bottom)
        margin: fraction of width/height considered off-screen.

    Returns:
        str: "in screen" or "outside of screen".
    """
    if norm_x is None or norm_y is None:
        return "no face/eyes detected"

    if norm_x < margin or norm_x > (1 - margin) or norm_y < margin or norm_y > (1 - margin):
        return "outside of screen"
    return "in screen"


def classify_gaze_by_offset(avg_dx, avg_dy, eye_dist, horiz_thresh=0.35, vert_thresh=0.35):
    """Classify gaze based on normalized iris offset relative to inter-ocular distance.

    This treats gaze direction relative to the face instead of where the projected
    gaze point falls on the camera frame, which prevents the classification from
    being triggered simply because the user's head is near the camera border.

    Args:
        avg_dx, avg_dy: average offset from eye center to iris center (normalized coords)
        eye_dist: distance between left and right eye centers (normalized coords)
        horiz_thresh: threshold (fraction of eye_dist) for horizontal "looking away"
        vert_thresh: threshold (fraction of eye_dist) for vertical "looking away"

    Returns:
        str: one of "in screen", "looking left", "looking right", "looking up", "looking down"
    """
    if avg_dx is None or avg_dy is None or eye_dist is None or eye_dist == 0:
        return "no face/eyes detected"

    # normalize offsets by eye distance to be robust to face size/zoom
    rel_dx = avg_dx / eye_dist
    rel_dy = avg_dy / eye_dist

    # Determine direction by thresholding the relative offsets.
    if abs(rel_dx) > abs(rel_dy):
        if rel_dx > horiz_thresh:
            return "looking right"
        if rel_dx < -horiz_thresh:
            return "looking left"
    else:
        if rel_dy > vert_thresh:
            return "looking down"
        if rel_dy < -vert_thresh:
            return "looking up"

    return "in screen"


def run_eye_tracker(src=0, no_display=False, projection_scale=4.0, invert=False, debug=False):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open video source: {src}")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # enables iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # MediaPipe iris landmark indices (refined face mesh)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # Approximate eye contour landmarks (used to compute an eye center / box).
    # These are commonly used MediaPipe landmark indices surrounding each eye.
    # We use them to compute the iris offset relative to the eye region and
    # project a gaze vector. These index sets are reasonable assumptions from
    # MediaPipe's face mesh mapping and work for typical models.
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263]

    last_print = 0.0
    print_interval = 5.0
    last_status = "no data"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from source")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            gaze_x = None
            gaze_y = None
            base_x = None
            base_y = None
            # face-relative classification (initialized per-frame)
            face_relative_status = None
            # nose landmark (initialized per-frame)
            nx = None
            ny = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                def mean_landmark_coords(indices):
                    xs = [landmarks[i].x for i in indices]
                    ys = [landmarks[i].y for i in indices]
                    return sum(xs) / len(xs), sum(ys) / len(ys)

                # initialize variables so static analyzers won't complain
                lx = ly = rx = ry = None
                lex = ley = rex = rey = None
                avg_dx = avg_dy = None
                base_x = base_y = None

                try:
                    # iris centers (normalized)
                    lx, ly = mean_landmark_coords(LEFT_IRIS)
                    rx, ry = mean_landmark_coords(RIGHT_IRIS)

                    # nose tip (normalized). MediaPipe examples commonly use
                    # landmark index 1 as the nose tip.
                    nx, ny = landmarks[1].x, landmarks[1].y

                    # compute eye contour centers (normalized)
                    lex, ley = mean_landmark_coords(LEFT_EYE_CONTOUR)
                    rex, rey = mean_landmark_coords(RIGHT_EYE_CONTOUR)

                    # For each eye compute the offset from eye center to iris center.
                    # This offset indicates where the iris lies within the eye box.
                    ldx, ldy = lx - lex, ly - ley
                    rdx, rdy = rx - rex, ry - rey

                    # Average the two eye offsets for a single gaze vector
                    avg_dx, avg_dy = (ldx + rdx) / 2.0, (ldy + rdy) / 2.0

                    # Base point to draw from: midpoint between both iris centers
                    base_x = (lx + rx) / 2.0
                    base_y = (ly + ry) / 2.0

                    # Optionally invert the direction (useful if offsets
                    # appear reversed for your camera/setup)
                    if invert:
                        avg_dx, avg_dy = -avg_dx, -avg_dy

                    # Project the gaze point outward from the face by scaling the
                    # offset. The scale controls how far the line extends on screen.
                    proj_x = base_x + avg_dx * projection_scale
                    proj_y = base_y + avg_dy * projection_scale

                    # Use these projected normalized coords as the gaze point used
                    # for classification and drawing. Clamp to [0,1]
                    gaze_x = max(0.0, min(1.0, proj_x))
                    gaze_y = max(0.0, min(1.0, proj_y))
                except Exception:
                    # If something unexpected with landmarks, fallback to None
                    gaze_x = None
                    gaze_y = None

                # Draw debug overlays: iris centers, eye contour centers,
                # per-eye offset vectors, and the projected gaze vector
                if base_x is not None and gaze_x is not None and gaze_y is not None and not no_display:
                    base_cx = int(base_x * w)
                    base_cy = int(base_y * h)
                    proj_cx = int(gaze_x * w)
                    proj_cy = int(gaze_y * h)

                    # draw iris centers (blue)
                    lx_c = int(lx * w)
                    ly_c = int(ly * h)
                    rx_c = int(rx * w)
                    ry_c = int(ry * h)
                    cv2.circle(frame, (lx_c, ly_c), 3, (255, 0, 0), -1)
                    cv2.circle(frame, (rx_c, ry_c), 3, (255, 0, 0), -1)

                    # draw eye contour centers (red)
                    lex_c = int(lex * w)
                    ley_c = int(ley * h)
                    rex_c = int(rex * w)
                    rey_c = int(rey * h)
                    cv2.circle(frame, (lex_c, ley_c), 3, (0, 0, 255), -1)
                    cv2.circle(frame, (rex_c, rey_c), 3, (0, 0, 255), -1)

                    # draw per-eye offset vectors (yellow)
                    cv2.line(frame, (lex_c, ley_c), (lx_c, ly_c), (0, 255, 255), 1)
                    cv2.line(frame, (rex_c, rey_c), (rx_c, ry_c), (0, 255, 255), 1)

                    # small circle at the eye midpoint (green) and projected line
                    # emphasize the base (face midpoint) with a larger green dot
                    cv2.circle(frame, (base_cx, base_cy), 8, (0, 255, 0), -1)
                    cv2.line(frame, (base_cx, base_cy), (proj_cx, proj_cy), (0, 255, 0), 2)

                    # debug text overlay
                    if debug:
                        info = f"dx={avg_dx:.3f} dy={avg_dy:.3f} scale={projection_scale}"
                        cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        # Also print to console occasionally (every 10 frames)
                        if int(time.time() * 10) % 10 == 0:
                            print(f"offsets: {info} base=({base_x:.3f},{base_y:.3f}) proj=({gaze_x:.3f},{gaze_y:.3f})")

                # Compute face bounding box (normalized) from all landmarks so
                # we can draw it and use it for compound face-orientation checks.
                bbox_min_x = min(l.x for l in landmarks)
                bbox_min_y = min(l.y for l in landmarks)
                bbox_max_x = max(l.x for l in landmarks)
                bbox_max_y = max(l.y for l in landmarks)

                # Draw the face bounding box (pixel coords) for visual feedback
                if not no_display:
                    x1 = int(bbox_min_x * w)
                    y1 = int(bbox_min_y * h)
                    x2 = int(bbox_max_x * w)
                    y2 = int(bbox_max_y * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, 'Face', (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # When landmarks are present compute a gaze classification that is
                # relative to the face using iris offsets. This avoids classifying
                # the user as "outside of screen" simply because their head nears
                # the camera border. We compute the inter-ocular distance and use
                # it to normalize the offset.
                face_relative_status = None
                # Compute face-relative classification only if all required
                # normalized coordinates are available.
                # Compound face-oriented check: prefer face bounding-box based
                # orientation (nose vs face center + symmetry) so the tracker
                # reasons about the whole face rather than only eyes.
                if lex is not None and ley is not None and rex is not None and rey is not None:
                    lx_f = float(lex)
                    ley_f = float(ley)
                    rx_f = float(rex)
                    rey_f = float(rey)
                    # eye distance in normalized coordinates
                    eye_dist = ((lx_f - rx_f) ** 2 + (ley_f - rey_f) ** 2) ** 0.5
                    if eye_dist > 0:
                        # Use full-face bbox center when possible
                        face_cx = (bbox_min_x + bbox_max_x) / 2.0
                        face_cy = (bbox_min_y + bbox_max_y) / 2.0
                        bbox_w = bbox_max_x - bbox_min_x
                        bbox_h = bbox_max_y - bbox_min_y

                        if nx is not None and ny is not None and bbox_w > 0 and bbox_h > 0:
                            rel_nose_dx = (float(nx) - face_cx) / bbox_w
                            rel_nose_dy = (float(ny) - face_cy) / bbox_h

                            # thresholds relative to bbox size: if nose stays
                            # near bbox center the face is facing the screen.
                            bbox_yaw_thresh = 0.20
                            bbox_pitch_thresh = 0.25

                            # symmetry check: nose should be centrally located
                            # between eyes when facing forward
                            left_dist = abs(float(nx) - lx_f)
                            right_dist = abs(rx_f - float(nx))
                            symmetry_rel = abs(left_dist - right_dist) / eye_dist if eye_dist > 0 else 1.0

                            if abs(rel_nose_dx) < bbox_yaw_thresh and abs(rel_nose_dy) < bbox_pitch_thresh and symmetry_rel < 0.45:
                                face_relative_status = "in screen"
                            else:
                                face_relative_status = "looking away"
                        else:
                            # Fallback to iris-based face-relative classification
                            if avg_dx is not None and avg_dy is not None:
                                face_relative_status = classify_gaze_by_offset(avg_dx, avg_dy, eye_dist)

            # Compute the current classification (used for overlay) and
            # also print it every print_interval seconds to the console.
            # Prefer face-relative classification when available; fall back to
            # the original projection-based classification otherwise.
            status = None
            if 'face_relative_status' in locals() and face_relative_status is not None:
                status = face_relative_status
            else:
                status = classify_gaze(gaze_x, gaze_y)
            last_status = status

            now = time.time()
            if now - last_print >= print_interval:
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
                print(f"[{ts}] gaze: {status}")
                last_print = now

            if not no_display:
                # Draw the current status on the frame so it's visible on the
                # camera window as well as printed to console.
                status_text = last_status
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                # position the status text under the top left area
                text_size, _ = cv2.getTextSize(status_text, font, font_scale, thickness)
                tx, ty = 10, 60
                # draw a filled rectangle as a background for readability
                cv2.rectangle(frame, (tx - 5, ty - text_size[1] - 5), (tx + text_size[0] + 5, ty + 5), (0, 0, 0), -1)
                cv2.putText(frame, status_text, (tx, ty), font, font_scale, (255, 255, 255), thickness)

                cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow('Eye Tracker (press q to quit)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Simple eye tracker using MediaPipe')
    parser.add_argument('--src', type=int, default=0, help='Video source (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Do not open display window')
    parser.add_argument('--scale', type=float, default=4.0, help='Projection scale for gaze vector')
    parser.add_argument('--invert', action='store_true', help='Invert projected direction (use if gaze appears reversed)')
    parser.add_argument('--debug', action='store_true', help='Show debug overlays and print offsets')
    args = parser.parse_args()

    run_eye_tracker(src=args.src, no_display=args.no_display, projection_scale=args.scale, invert=args.invert, debug=args.debug)


if __name__ == '__main__':
    main()
