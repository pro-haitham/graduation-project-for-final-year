import os
import csv
import time
from datetime import datetime
import cv2
from src.camera_processor import CameraProcessor
from src.recognition_engine import RecognitionEngine

LOG_FILE = "logs/week1_evaluation_results.csv"

def init_csv_logger() -> None:
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Test_Condition", "Expected_Identity",
                "Predicted_Identity", "Cosine_Score", "Brightness",
                "Sharpness", "Latency_ms", "Outcome"
            ])

def run_detection_test(cam: CameraProcessor) -> None:
    if not cam.start_camera():
        print("[ERROR] Could not open webcam.")
        return

    print("\n--- DAY 1-3: CAMERA & FACE QUALITY TEST ---")
    print("Click inside the video window and press 'q' to return to the menu.")

    try:
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                print("[ERROR] Failed to read frame from camera.")
                break

            faces = cam.detect_faces(frame)
            if len(faces) == 0:
                annotated = cam.draw_feedback(frame, faces, "No Face Detected", (0, 0, 255))
            elif len(faces) > 1:
                annotated = cam.draw_feedback(
                    frame, faces, f"Multiple Faces ({len(faces)}) - Only 1 Allowed", (0, 165, 255)
                )
            else:
                crop = cam.extract_face_crop(frame, faces[0])
                quality = cam.check_image_quality(crop)
                color = (0, 255, 0) if quality["passed"] else (0, 165, 255)
                status = (
                    f"{quality['reason']} | Bright:{quality['brightness']:.0f} "
                    f"Sharp:{quality['sharpness']:.0f}"
                )
                annotated = cam.draw_feedback(frame, faces, status, color)
                annotated[45:157, annotated.shape[1]-117:annotated.shape[1]-5] = crop

            cv2.imshow("Day 1-3: Face Detection & Processing", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()

def run_user_registration(cam: CameraProcessor, engine: RecognitionEngine) -> None:
    print("\n--- DAY 4: USER REGISTRATION ---")
    username = input("Enter new username to enroll: ").strip()
    if not username:
        print("[ABORTED] Username cannot be empty.")
        return

    consent = input(f"Does '{username}' consent to biometric enrollment? (yes/no): ").strip().lower()
    if consent not in ("yes", "y"):
        print("[ABORTED] Explicit user consent is required.")
        return

    if not cam.start_camera():
        print("[ERROR] Could not open webcam.")
        return

    prompts = [
        "Sample 1/5: Look straight at camera",
        "Sample 2/5: Turn head slightly LEFT",
        "Sample 3/5: Turn head slightly RIGHT",
        "Sample 4/5: Tilt head slightly UP",
        "Sample 5/5: Normal expression, straight"
    ]
    captured_embeddings = []

    print("Press SPACEBAR in the video window when the box is GREEN. Press 'q' to cancel.")

    try:
        while len(captured_embeddings) < 5:
            ok, frame = cam.read_frame()
            if not ok:
                break

            faces = cam.detect_faces(frame)
            current_prompt = prompts[len(captured_embeddings)]

            if len(faces) != 1:
                msg = "Show 1 face" if len(faces) == 0 else "Multiple faces!"
                annotated = cam.draw_feedback(frame, faces, f"{current_prompt} [{msg}]", (0, 0, 255))
                can_capture = False
            else:
                crop = cam.extract_face_crop(frame, faces[0])
                quality = cam.check_image_quality(crop)
                can_capture = quality["passed"]
                color = (0, 255, 0) if can_capture else (0, 165, 255)
                annotated = cam.draw_feedback(
                    frame, faces, f"{current_prompt} [{quality['reason']}]", color
                )

            cv2.imshow("Day 4: User Enrollment", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[CANCELLED] Enrollment cancelled.")
                return
            elif key == ord(" ") and can_capture:
                emb = engine.extract_embedding(frame, faces[0])
                captured_embeddings.append(emb)
                print(f"[CAPTURED] Sample {len(captured_embeddings)}/5 saved.")
                time.sleep(0.3)

        user_id = engine.register_user(username, captured_embeddings, consent_given=True)
        print(f"\n[SUCCESS] User '{username}' registered with Derived ID: {user_id}")
    finally:
        cam.release()

def run_recognition_and_testing(cam: CameraProcessor, engine: RecognitionEngine) -> None:
    init_csv_logger()
    print("\n--- DAY 5 & 6: FACE RECOGNITION & EVALUATION ---")
    condition = input("Enter test condition (e.g., Normal_Light, Low_Light, Angle_30deg, Unregistered): ").strip() or "Normal_Light"
    expected_user = input("Enter expected identity name (or 'Unknown' for unregistered test): ").strip() or "Unknown"

    if not cam.start_camera():
        print("[ERROR] Could not open webcam.")
        return

    print("Press 'l' in the video window to log trial to CSV | Press 'q' to exit.")

    try:
        while True:
            t_start = time.perf_counter()
            ok, frame = cam.read_frame()
            if not ok:
                break

            faces = cam.detect_faces(frame)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            if len(faces) == 1:
                crop = cam.extract_face_crop(frame, faces[0])
                quality = cam.check_image_quality(crop)
                emb = engine.extract_embedding(frame, faces[0])
                matched, identity, score = engine.verify_identity(emb)
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                color = (0, 255, 0) if matched else (0, 0, 255)
                status = f"ID: {identity} ({score:.2f}) | {latency_ms:.1f}ms | [Press 'L' to Log]"
                annotated = cam.draw_feedback(frame, faces, status, color)
            elif len(faces) == 0:
                annotated = cam.draw_feedback(frame, faces, "Waiting for face...", (0, 0, 255))
                identity, score, quality = "No_Face", 0.0, {"brightness": 0.0, "sharpness": 0.0}
            else:
                annotated = cam.draw_feedback(frame, faces, "Security Block: Multiple Faces!", (0, 165, 255))
                identity, score, quality = "Multiple_Faces", 0.0, {"brightness": 0.0, "sharpness": 0.0}

            cv2.imshow("Day 5-6: Face Recognition & Evaluation", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("l") and len(faces) == 1:
                outcome = "PASS" if identity.lower() == expected_user.lower() else "FAIL"
                with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        condition,
                        expected_user,
                        identity,
                        f"{score:.4f}",
                        f"{quality['brightness']:.1f}",
                        f"{quality['sharpness']:.1f}",
                        f"{latency_ms:.2f}",
                        outcome
                    ])
                print(f"[LOGGED] Condition={condition} | Expected={expected_user} | Got={identity} ({score:.3f}) -> {outcome}")
    finally:
        cam.release()

def main() -> None:
    cam = CameraProcessor()
    engine = RecognitionEngine()

    while True:
        print("\n========================================================")
        print(" WEEK 1: ANTI-SPOOFING FACIAL AUTHENTICATION FOUNDATION")
        print("========================================================")
        print(f" Enrolled Users Currently Loaded: {len(engine.enrolled_templates)}")
        print(" 1. [Days 1-3] Run Camera, Face Detection & Quality Test")
        print(" 2. [Day 4]    Register / Enroll a New User (5 Angles)")
        print(" 3. [Days 5-6] Run Face Recognition & Log Test Results")
        print(" 4. [Exit]     Quit Application")
        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            run_detection_test(cam)
        elif choice == "2":
            run_user_registration(cam, engine)
        elif choice == "3":
            run_recognition_and_testing(cam, engine)
        elif choice == "4":
            print("Exiting application.")
            break
        else:
            print("Invalid option. Please choose 1-4.")

if __name__ == "__main__":
    main()
