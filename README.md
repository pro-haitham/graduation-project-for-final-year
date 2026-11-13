# Anti-Spoofing Facial Authentication System (Days 1-22)

## Overview
Multi-layered biometric authentication combining YuNet Face Detection, MobileNetV3-Style Passive FAS (Spatial/Chromatic/2D-FFT), Randomized 3D Active Liveness, and SFace Recognition.

## Phase 1-4 Verified Deliverables
- **Phase 1 (Days 1-7):** YuNet 5-point landmark detection, quality filtering, multi-sample enrollment, and SFace cosine verification (threshold=0.42).
- **Phase 2 (Days 8-10):** Real/Spoof dataset pipeline (live, spoof_print, spoof_replay), video frame extraction, and leak-free Train/Val/Test split (data/processed_dataset/dataset_manifest.json).
- **Phase 3 (Days 11-17):** Two-stage training (Frozen Head -> Unfrozen Fine-Tuning). Unseen Test Set ISO/IEC 30107-3 Metrics: Accuracy: 87.5%, APCER: 0.0000, BPCER: 0.2500, ACER: 0.1250 (logs/phase3_fas_evaluation.json).
- **Phase 4 (Days 18-22):** 3D Head-Turn, Eyeball-Slit Blink, and Multi-Cue Smile active challenges integrated with 12-frame rolling passive score aggregation (logs/phase4_integrated_auth_log.csv).

## How to Run
- Interactive App: python main.py
- Automated Audit (Days 8-22): python verify_days_8_to_22.py
