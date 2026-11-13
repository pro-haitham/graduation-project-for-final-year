import os
import csv
import time
import random
from datetime import datetime
import cv2
import numpy as np
from src.camera_processor import CameraProcessor
from src.recognition_engine import RecognitionEngine
from src.dataset_manager import DatasetManager
from src.passive_fas_model import PassiveFASModel
from src.active_liveness import ActiveLivenessVerifier

INTEGRATED_LOG = "logs/phase4_integrated_auth_log.csv"

def init_integrated_logger() -> None:
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(INTEGRATED_LOG):
        with open(INTEGRATED_LOG, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Active_Challenges", "Active_Passed",
                "Avg_Passive_Score", "Passive_Passed", "Predicted_Identity",
                "Cosine_Score", "Final_Decision"
            ])

def run_phase2_data_pipeline(cam: CameraProcessor, ds_manager: DatasetManager) -> None:
    print("\n--- PHASE 2 (DAYS 8-10): AI DATA PREPARATION & SPLITTING ---")
    print(" 1. Capture new Real/Spoof samples via Webcam")
    print(" 2. Run Data Integrity Filter & Subject-Disjoint Split (Train/Val/Test)")
    sub = input("Select (1 or 2): ").strip()

    if sub == "1":
        subj = input("Enter Subject ID (e.g., solo, aha, S01): ").strip() or "S01"
        print("Select Category:")
        print("  1. live         (Real Genuine Face)")
        print("  2. spoof_print  (Printed Photo / ID Card Attack)")
        print("  3. spoof_replay (Phone / Screen Video Replay Attack)")
        cat_choice = input("Enter 1, 2, or 3: ").strip()
        cat_map = {"1": "live", "2": "spoof_print", "3": "spoof_replay",
                   "live": "live", "spoof_print": "spoof_print", "spoof_replay": "spoof_replay"}
        cat = cat_map.get(cat_choice, "")
        if not cat:
            print("[ERROR] Invalid category choice.")
            return
        ds_manager.collect_webcam_samples(cam, subject_id=subj, category=cat, num_samples=25)

    stats = ds_manager.prepare_and_split_dataset()
    print("\n[PHASE 2 DELIVERABLE: DATASET MANIFEST]")
    print(f" - Data Leakage Check : {stats['data_leakage_check']}")
    print(f" - Corrupted Removed  : {stats['corrupted_removed']}")
    print(f" - Train Split        : Live={stats['train']['live']}, Spoof={stats['train']['spoof']} (Subjects: {stats['train']['subjects']})")
    print(f" - Validation Split   : Live={stats['val']['live']}, Spoof={stats['val']['spoof']} (Subjects: {stats['val']['subjects']})")
    print(f" - Test Split         : Live={stats['test']['live']}, Spoof={stats['test']['spoof']} (Subjects: {stats['test']['subjects']})")
    print(" - Saved Manifest     : data/processed_dataset/dataset_manifest.json")

def run_phase3_training(fas_model: PassiveFASModel, ds_manager: DatasetManager) -> None:
    print("\n--- PHASE 3 (DAYS 11-17): TRAINING & EVALUATING ANTI-SPOOFING MODEL ---")
    ds_manager.prepare_and_split_dataset()
    report = fas_model.train_and_evaluate()
    m = report["metrics_iso_30107_3"]
    print("\n[PHASE 3 DELIVERABLE: ISO/IEC 30107-3 EVALUATION REPORT]")
    print(f" - Model Architecture   : {report['architecture']}")
    if "day15_stage_comparison" in report:
        c = report["day15_stage_comparison"]
        print(f" - Stage 1 (Frozen Head): Val Loss={c['Stage1_Frozen_HeadOnly']['best_val_loss']} | Val Acc={c['Stage1_Frozen_HeadOnly']['best_val_acc']*100:.1f}%")
        print(f" - Stage 2 (Fine-Tuned) : Val Loss={c['Stage2_Unfrozen_FineTuned']['best_val_loss']} | Val Acc={c['Stage2_Unfrozen_FineTuned']['best_val_acc']*100:.1f}%")
        print(f" - Retained Best Version: {c['retained_version']}")
    print(f" - Test Set Accuracy    : {m['Test_Accuracy'] * 100:.2f}%")
    print(f" - APCER (Overall)      : {m['APCER_Overall']:.4f}")
    print(f" - APCER (By Attack)    : {m['APCER_By_Attack']}")
    print(f" - BPCER (Bona Fide Err): {m['BPCER']:.4f}")
    print(f" - ACER (Average Error) : {m['ACER']:.4f}")
    print(f" - Model Reload Verified: {report['model_reload_verified']}")
    print(" - Weights Saved To     : models/passive_fas_weights.npz")
    print(" - Full Report Saved To : logs/phase3_fas_evaluation.json")

def run_phase4_landmark_and_active_test(cam: CameraProcessor, active_verifier: ActiveLivenessVerifier) -> None:
    if not cam.start_camera():
        print("[ERROR] Could not open webcam.")
        return

    active_verifier.reset_session()
    challenges = random.sample(active_verifier.CHALLENGE_POOL, 3)
    current_idx = 0
    stage_start = time.time()

    print("\n--- PHASE 4 (DAYS 18-20): LANDMARK TRACKING & RANDOMIZED CHALLENGES ---")
    print("Complete the 2 random on-screen movement challenges. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                break

            faces = cam.detect_faces(frame)
            if len(faces) != 1:
                annotated = cam.draw_feedback(
                    frame, faces, "Face Lost! Position 1 face in center", (0, 0, 255)
                )
            else:
                pose = active_verifier.analyze_pose_and_expression(frame, faces[0])
                if active_verifier.baseline_mouth_ratio is None:
                    active_verifier.baseline_mouth_ratio = pose["smile_ratio"]
                    active_verifier.baseline_pitch_ratio = pose["pitch_ratio"]
                    active_verifier.baseline_eye_openness = pose["eye_openness"]

                if current_idx < len(challenges):
                    code, prompt_text = challenges[current_idx]
                    elapsed = time.time() - stage_start
                    remaining = max(0.0, active_verifier.challenge_timeout_sec - elapsed)

                    if active_verifier.check_challenge_satisfied(code, pose):
                        print(f"[CHALLENGE PASSED] {prompt_text}")
                        current_idx += 1
                        stage_start = time.time()
                        time.sleep(0.4)
                    elif remaining <= 0.0:
                        print(f"[TIMEOUT] Challenge '{prompt_text}' timed out! Resetting random challenge...")
                        challenges = random.sample(active_verifier.CHALLENGE_POOL, 3)
                        current_idx = 0
                        stage_start = time.time()

                    status = f"[{current_idx+1}/3] {prompt_text} ({remaining:.1f}s) | Yaw:{pose['yaw_ratio']:.2f} Smile:{pose['smile_ratio']:.2f}"
                    annotated = cam.draw_feedback(frame, faces, status, (0, 255, 255))
                else:
                    annotated = cam.draw_feedback(
                        frame, faces, "ACTIVE LIVENESS VERIFIED! (Press 'q' to exit)", (0, 255, 0)
                    )

            cv2.imshow("Days 18-20: Active Liveness & Landmark Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()

def run_day22_full_integrated_authentication(
    cam: CameraProcessor,
    engine: RecognitionEngine,
    fas_model: PassiveFASModel,
    active_verifier: ActiveLivenessVerifier
) -> None:
    init_integrated_logger()
    if not fas_model.is_trained:
        print("[INFO] Passive FAS model not yet trained. Training automatically now...")
        ds = DatasetManager()
        ds.prepare_and_split_dataset()
        fas_model.train_and_evaluate()

    if not cam.start_camera():
        print("[ERROR] Could not open webcam.")
        return

    active_verifier.reset_session()
    challenges = random.sample(active_verifier.CHALLENGE_POOL, 3)
    challenge_names = "+".join(c[0] for c in challenges)
    current_idx = 0
    retries_used = 0
    stage_start = time.time()
    decision_finalized = False
    final_banner = ""
    final_color = (0, 255, 0)

    # Store the most frontal face frame & face_row so sideways head turns don't hurt ID matching
    best_frontal_frame = None
    best_frontal_face_row = None
    min_abs_yaw = 999.0

    print("\n--- DAY 22: INTEGRATED MULTI-MODAL AUTHENTICATION PIPELINE ---")
    print("Pipeline: Face Detection -> Multi-Frame Passive FAS + Active Challenges -> SFace Identity Verification")

    try:
        while True:
            ok, frame = cam.read_frame()
            if not ok:
                break

            faces = cam.detect_faces(frame)

            if decision_finalized:
                annotated = cam.draw_feedback(frame, faces, final_banner, final_color)
                cv2.imshow("Day 22: Integrated Anti-Spoofing Authentication", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if len(faces) != 1:
                msg = "Waiting for 1 Face..." if len(faces) == 0 else "SECURITY BLOCK: Multiple Faces!"
                annotated = cam.draw_feedback(frame, faces, msg, (0, 0, 255))
            else:
                face_row = faces[0]
                crop = cam.extract_face_crop(frame, face_row, target_size=(128, 128))
                quality = cam.check_image_quality(crop)

                pose = active_verifier.analyze_pose_and_expression(frame, face_row)
                frame_fas_prob = fas_model.predict_frame(crop, full_frame=frame, face_row=face_row)
                if abs(pose["yaw_ratio"]) < 0.15 and quality["passed"]:
                    avg_passive_score = active_verifier.add_passive_score(frame_fas_prob)
                elif len(active_verifier.passive_scores) > 0:
                    avg_passive_score = sum(active_verifier.passive_scores) / len(active_verifier.passive_scores)
                else:
                    avg_passive_score = frame_fas_prob
                if active_verifier.baseline_mouth_ratio is None:
                    active_verifier.baseline_mouth_ratio = pose["smile_ratio"]
                    active_verifier.baseline_pitch_ratio = pose["pitch_ratio"]
                    active_verifier.baseline_eye_openness = pose["eye_openness"]

                # Save cleanest frontal frame (lowest yaw angle) for accurate SFace recognition
                if quality["passed"] and abs(pose["yaw_ratio"]) < min_abs_yaw:
                    min_abs_yaw = abs(pose["yaw_ratio"])
                    best_frontal_frame = frame.copy()
                    best_frontal_face_row = face_row.copy()

                if current_idx < len(challenges):
                    code, prompt_text = challenges[current_idx]
                    if active_verifier.waiting_for_neutral:
                        prompt_text = "Return Face to CENTER (Look Straight)"
                    remaining = max(0.0, active_verifier.challenge_timeout_sec - (time.time() - stage_start))

                    if active_verifier.replay_desync_detected:
                        decision_finalized = True
                        final_banner = f"ACCESS DENIED: Video Replay Desync Detected! (Wrong Action) [Press 'q']"
                        final_color = (0, 0, 255)
                        print(f"\n[FINAL DECISION] {final_banner}")
                        continue

                    if active_verifier.check_challenge_satisfied(code, pose):
                        print(f"[OK] Completed {prompt_text}")
                        current_idx += 1
                        stage_start = time.time()
                    elif remaining <= 0.0:
                        retries_used += 1
                        if retries_used > active_verifier.max_retries:
                            decision_finalized = True
                            final_banner = f"ACCESS DENIED: Spoof / Challenge Failed (Passive:{avg_passive_score:.2f}) [Press 'q']"
                            final_color = (0, 0, 255)
                            print(f"\n[FINAL DECISION] {final_banner}")
                            with open(INTEGRATED_LOG, mode="a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    challenge_names, False, f"{avg_passive_score:.3f}",
                                    avg_passive_score >= fas_model.liveness_threshold,
                                    "N/A", "0.000", "DENIED_ACTIVE_LIVENESS"
                                ])
                            continue
                        else:
                            print(f"[RETRY {retries_used}/{active_verifier.max_retries}] New random challenge issued.")
                            challenges = random.sample(active_verifier.CHALLENGE_POOL, 3)
                            current_idx = 0
                            stage_start = time.time()

                    status = f"Passive:{avg_passive_score:.2f} | [{current_idx+1}/3] {prompt_text} ({remaining:.1f}s)"
                    annotated = cam.draw_feedback(frame, faces, status, (0, 255, 255))
                else:
                    passive_passed = avg_passive_score >= 0.42
                    ref_frame = best_frontal_frame if best_frontal_frame is not None else frame
                    ref_face = best_frontal_face_row if best_frontal_face_row is not None else face_row

                    emb = engine.extract_embedding(ref_frame, ref_face)
                    id_matched, username, cos_score = engine.verify_identity(emb)

                    if passive_passed and id_matched:
                        final_decision = "ACCESS_GRANTED"
                        final_banner = f"ACCESS GRANTED: Welcome {username}! (Live:{avg_passive_score:.2f} ID:{cos_score:.2f}) [Press 'q']"
                        final_color = (0, 255, 0)
                    elif not passive_passed:
                        final_decision = "DENIED_PASSIVE_SPOOF"
                        final_banner = f"ACCESS DENIED: Passive Spoof Detected ({avg_passive_score:.2f}) [Press 'q']"
                        final_color = (0, 0, 255)
                    else:
                        final_decision = "DENIED_UNKNOWN_IDENTITY"
                        final_banner = f"ACCESS DENIED: Live Face but Unknown User ({cos_score:.2f}) [Press 'q']"
                        final_color = (0, 0, 255)

                    decision_finalized = True
                    print(f"\n[FINAL DECISION] {final_banner}")
                    with open(INTEGRATED_LOG, mode="a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            challenge_names, True, f"{avg_passive_score:.3f}",
                            passive_passed, username, f"{cos_score:.4f}", final_decision
                        ])
                    annotated = cam.draw_feedback(frame, faces, final_banner, final_color)

            cv2.imshow("Day 22: Integrated Anti-Spoofing Authentication", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()

def main() -> None:
    cam = CameraProcessor()
    engine = RecognitionEngine()
    ds_manager = DatasetManager()
    fas_model = PassiveFASModel()
    active_verifier = ActiveLivenessVerifier()

    while True:
        print("\n=====================================================================")
        print(" ANTI-SPOOFING FACIAL AUTHENTICATION SYSTEM (PHASES 1 TO 4: DAYS 1-22)")
        print("=====================================================================")
        print(f" Enrolled Users: {len(engine.enrolled_templates)} | Passive FAS Model Trained: {fas_model.is_trained}")
        print(" 1. [Phase 1: Days 1-7]   Enroll User / Baseline Setup")
        print(" 2. [Phase 2: Days 8-10]  Prepare AI Dataset & Subject-Disjoint Split")
        print(" 3. [Phase 3: Days 11-17] Train & Evaluate Passive Anti-Spoofing Model (APCER/BPCER/ACER)")
        print(" 4. [Phase 4: Days 18-20] Test Landmark Tracking & Randomized Active Challenges")
        print(" 5. [Phase 4: Day 22]     RUN FULL INTEGRATED AUTHENTICATION (Liveness + Identity)")
        print(" 6. [Exit]                Quit Application")
        choice = input("\nSelect an option (1-6): ").strip()

        if choice == "1":
            username = input("Enter username to enroll (or press Enter to skip): ").strip()
            if username:
                consent = input(f"Does '{username}' consent? (yes/no): ").strip().lower()
                if consent in ("yes", "y") and cam.start_camera():
                    embs = []
                    print("Press SPACEBAR 5 times in the camera window...")
                    try:
                        while len(embs) < 5:
                            ok, frame = cam.read_frame()
                            if not ok:
                                break
                            faces = cam.detect_faces(frame)
                            annotated = cam.draw_feedback(frame, faces, f"Capture {len(embs)+1}/5 (Press SPACE)", (0, 255, 0))
                            cv2.imshow("Enroll", annotated)
                            k = cv2.waitKey(1) & 0xFF
                            if k == ord(" ") and len(faces) == 1:
                                embs.append(engine.extract_embedding(frame, faces[0]))
                                print(f"Captured {len(embs)}/5")
                            elif k == ord("q"):
                                break
                        if len(embs) == 5:
                            uid = engine.register_user(username, embs, True)
                            print(f"[SUCCESS] Enrolled {username} ({uid})")
                    finally:
                        cam.release()
        elif choice == "2":
            run_phase2_data_pipeline(cam, ds_manager)
        elif choice == "3":
            run_phase3_training(fas_model, ds_manager)
        elif choice == "4":
            run_phase4_landmark_and_active_test(cam, active_verifier)
        elif choice == "5":
            run_day22_full_integrated_authentication(cam, engine, fas_model, active_verifier)
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
