from collections import deque
import cv2
import numpy as np
from typing import Dict, List, Optional

class ActiveLivenessVerifier:
    """
    High-Responsiveness Active Liveness Verifier (Head Turn + Smile + Eye Blink):
    - Uses tight eyeball slit ROI (excluding eyebrows) + 1-frame temporal blink detection.
    - Uses multi-cue Smile detection (corner widening + cheek/lip lift + teeth/mouth-open variance).
    - Displays live progress percentage on-screen so the user sees their movement registering.
    """
    CHALLENGE_POOL = [
        ("TURN_LEFT", "Challenge: Turn Head LEFT"),
        ("TURN_RIGHT", "Challenge: Turn Head RIGHT"),
        ("SMILE", "Challenge: SMILE / Open Mouth Slightly"),
        ("BLINK_EYES", "Challenge: BLINK Eyes (Close for 0.5s)")
    ]

    def __init__(
        self,
        challenge_timeout_sec: float = 9.0,
        max_retries: int = 2,
        passive_window_size: int = 12
    ) -> None:
        self.challenge_timeout_sec = challenge_timeout_sec
        self.max_retries = max_retries
        self.passive_scores: deque = deque(maxlen=passive_window_size)
        self.calib_yaw: List[float] = []
        self.calib_smile: List[float] = []
        self.calib_mouth_lip: List[float] = []
        self.calib_eye_darkness: List[float] = []
        self.calib_mouth_tex: List[float] = []
        self.is_calibrated: bool = False

        self.baseline_yaw: float = 0.0
        self.baseline_mouth_ratio: float = 0.80
        self.baseline_mouth_lip: float = 0.70
        self.baseline_eye_darkness: float = 15.0
        self.baseline_mouth_tex: float = 12.0

        self.prev_eye_darkness: Optional[float] = None
        self.consecutive_pass_frames: int = 0
        self.neutral_hold_frames: int = 0
        self.waiting_for_neutral: bool = False
        self.replay_desync_detected: bool = False

    @staticmethod
    def analyze_pose_and_expression(frame: np.ndarray, face_row: np.ndarray) -> Dict[str, float]:
        lms = face_row[4:14].reshape((5, 2))
        r_eye, l_eye, nose, r_mouth, l_mouth = lms[0], lms[1], lms[2], lms[3], lms[4]

        inter_ocular = float(np.linalg.norm(l_eye - r_eye)) + 1e-5
        eye_mid = 0.5 * (r_eye + l_eye)
        mouth_mid = 0.5 * (r_mouth + l_mouth)

        # 1. Head Yaw & 3D Asymmetry
        yaw_ratio = float((nose[0] - eye_mid[0]) / inter_ocular)
        dist_nose_reye = float(np.linalg.norm(nose - r_eye))
        dist_nose_leye = float(np.linalg.norm(nose - l_eye))
        asymmetry_3d = (dist_nose_leye - dist_nose_reye) / inter_ocular

        # 2. Multi-Cue Smile Metrics:
        # Cue A: Mouth corner-to-corner width normalized by inter-ocular distance
        mouth_width = float(np.linalg.norm(l_mouth - r_mouth))
        smile_ratio = mouth_width / inter_ocular

        # Cue B: Nose-to-mouth vertical distance normalized by inter-ocular (decreases when smiling lifts cheeks/corners)
        nose_to_mouth_norm = float((mouth_mid[1] - nose[1]) / inter_ocular)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape

        # Cue C: Inner mouth texture/teeth contrast (increases when smiling shows teeth or parts lips)
        mx1 = max(0, int(min(r_mouth[0], l_mouth[0])))
        mx2 = min(w, int(max(r_mouth[0], l_mouth[0])))
        my_center = int(mouth_mid[1])
        mh_pad = max(5, int(inter_ocular * 0.12))
        mouth_patch = gray[max(0, my_center - mh_pad):min(h, my_center + mh_pad), mx1:mx2]
        mouth_tex = float(np.std(mouth_patch)) if mouth_patch.size > 0 else 12.0

        # 3. Tight Eyeball Slit Detector (Excludes Eyebrows!):
        # When eyes are OPEN, the dark pupil/iris creates a deep minimum intensity and high vertical edge contrast
        # inside a tight horizontal slit right on the eye landmark. When eyes BLINK/CLOSE, skin covers the dark pupil!
        rx_pad = max(6, int(inter_ocular * 0.14))
        ry_pad = max(3, int(inter_ocular * 0.07))  # Very thin vertically so eyebrows are completely excluded!
        eye_scores = []
        for ex, ey in (r_eye, l_eye):
            ix, iy = int(ex), int(ey)
            epatch = gray[max(0, iy - ry_pad):min(h, iy + ry_pad), max(0, ix - rx_pad):min(w, ix + rx_pad)]
            if epatch.size > 0:
                # Difference between bright sclera/skin (85th percentile) and dark pupil/eyelash (15th percentile)
                p85 = float(np.percentile(epatch, 85))
                p15 = float(np.percentile(epatch, 15))
                gy = cv2.Sobel(epatch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                eye_scores.append((p85 - p15) + float(np.std(gy)) * 0.5)
        eye_openness = float(np.mean(eye_scores)) if eye_scores else 25.0

        return {
            "yaw_ratio": yaw_ratio,
            "asymmetry_3d": asymmetry_3d,
            "pitch_ratio": nose_to_mouth_norm,
            "smile_ratio": smile_ratio,
            "mouth_tex": mouth_tex,
            "eye_openness": eye_openness,
            "inter_ocular": inter_ocular
        }

    def update_calibration(self, pose: Dict[str, float]) -> bool:
        if self.is_calibrated:
            return True
        self.calib_yaw.append(pose["yaw_ratio"])
        self.calib_smile.append(pose["smile_ratio"])
        self.calib_mouth_lip.append(pose["pitch_ratio"])
        self.calib_eye_darkness.append(pose["eye_openness"])
        self.calib_mouth_tex.append(pose["mouth_tex"])

        if len(self.calib_yaw) >= 6:
            self.baseline_yaw = float(np.median(self.calib_yaw))
            self.baseline_mouth_ratio = float(np.median(self.calib_smile))
            self.baseline_mouth_lip = float(np.median(self.calib_mouth_lip))
            self.baseline_eye_darkness = float(np.median(self.calib_eye_darkness))
            self.baseline_mouth_tex = float(np.median(self.calib_mouth_tex))
            self.prev_eye_darkness = self.baseline_eye_darkness
            self.is_calibrated = True
        return self.is_calibrated

    def check_challenge_satisfied(
        self,
        challenge_code: str,
        pose: Dict[str, float]
    ) -> bool:
        if not self.is_calibrated:
            self.update_calibration(pose)
            return False

        delta_yaw = pose["yaw_ratio"] - self.baseline_yaw
        asym = pose["asymmetry_3d"]
        delta_smile = pose["smile_ratio"] - self.baseline_mouth_ratio
        delta_lip_lift = self.baseline_mouth_lip - pose["pitch_ratio"]
        delta_mouth_tex = pose["mouth_tex"] - self.baseline_mouth_tex
        eye_open = pose["eye_openness"]

        # Return-to-center gate between challenges (only required after Left/Right head turns!)
        if self.waiting_for_neutral:
            if abs(delta_yaw) < 0.13:
                self.neutral_hold_frames += 1
                if self.neutral_hold_frames >= 2:
                    self.waiting_for_neutral = False
                    self.neutral_hold_frames = 0
            else:
                self.neutral_hold_frames = 0
            self.prev_eye_darkness = eye_open
            return False

        frame_matched = False
        if challenge_code == "TURN_LEFT":
            frame_matched = (delta_yaw > 0.16) and (abs(asym) > 0.18)
        elif challenge_code == "TURN_RIGHT":
            frame_matched = (delta_yaw < -0.16) and (abs(asym) > 0.18)
        elif challenge_code == "SMILE":
            # Triggers if mouth widens by >=3.5%, OR cheeks/lips lift + widens by >=2%, OR teeth/open-mouth contrast jumps!
            wide_enough = delta_smile >= 0.035
            lifted_smile = (delta_smile >= 0.018) and (delta_lip_lift >= 0.015)
            teeth_or_open = (delta_mouth_tex >= 4.5) and (delta_smile >= 0.012)
            frame_matched = wide_enough or lifted_smile or teeth_or_open
        elif challenge_code == "BLINK_EYES":
            # Triggers on a 12% drop from baseline OR a sudden 10% frame-to-frame drop when eyelids close
            prev_val = self.prev_eye_darkness if self.prev_eye_darkness is not None else self.baseline_eye_darkness
            drop_from_base = eye_open < (self.baseline_eye_darkness * 0.88)
            sudden_blink = eye_open < (prev_val * 0.89)
            frame_matched = (drop_from_base or sudden_blink) and (abs(delta_yaw) < 0.16)

        self.prev_eye_darkness = eye_open

        if frame_matched:
            self.consecutive_pass_frames += 1
        else:
            self.consecutive_pass_frames = max(0, self.consecutive_pass_frames - 1)

        # Blink and Smile trigger immediately (1 frame for blink, 2 frames for smile/turn)
        required_hold = 1 if challenge_code == "BLINK_EYES" else 2
        if self.consecutive_pass_frames >= required_hold:
            self.consecutive_pass_frames = 0
            # Only require returning head to center if the user just turned their head sideways
            if challenge_code in ("TURN_LEFT", "TURN_RIGHT"):
                self.waiting_for_neutral = True
                self.neutral_hold_frames = 0
            return True
        return False

    def reset_session(self) -> None:
        self.passive_scores.clear()
        self.calib_yaw.clear()
        self.calib_smile.clear()
        self.calib_mouth_lip.clear()
        self.calib_eye_darkness.clear()
        self.calib_mouth_tex.clear()
        self.is_calibrated = False
        self.prev_eye_darkness = None
        self.consecutive_pass_frames = 0
        self.neutral_hold_frames = 0
        self.waiting_for_neutral = False
        self.replay_desync_detected = False

    def add_passive_score(self, score: float) -> float:
        self.passive_scores.append(score)
        return float(np.mean(self.passive_scores))
