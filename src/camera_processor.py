import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

class CameraProcessor:
    """
    Handles video capture, YuNet DNN face detection, image quality verification
    (lighting & sharpness), and standardized face alignment/cropping.
    """
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        detector_model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.85
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

        if not os.path.exists(detector_model_path):
            raise FileNotFoundError(
                f"Detector model missing at '{detector_model_path}'."
            )

        self.detector = cv2.FaceDetectorYN.create(
            model=detector_model_path,
            config="",
            input_size=(self.width, self.height),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def start_camera(self) -> bool:
        """Initializes the camera safely with DirectShow on Windows."""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads and validates a single frame from the video stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        return True, frame

    def detect_faces(self, frame: np.ndarray) -> np.ndarray:
        """Detects faces in the frame using YuNet."""
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    @staticmethod
    def check_image_quality(face_crop: np.ndarray) -> Dict[str, Any]:
        """Evaluates lighting (brightness) and sharpness (Laplacian variance)."""
        if face_crop is None or face_crop.size == 0:
            return {"passed": False, "reason": "Empty face region", "brightness": 0.0, "sharpness": 0.0}

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 45.0:
            return {"passed": False, "reason": "Too Dark (Low Light)", "brightness": brightness, "sharpness": sharpness}
        if brightness > 220.0:
            return {"passed": False, "reason": "Too Bright (Overexposed)", "brightness": brightness, "sharpness": sharpness}
        if sharpness < 40.0:
            return {"passed": False, "reason": "Blurry (Hold Still)", "brightness": brightness, "sharpness": sharpness}

        return {"passed": True, "reason": "Quality OK", "brightness": brightness, "sharpness": sharpness}

    @staticmethod
    def extract_face_crop(frame: np.ndarray, face_row: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """Extracts and standardizes the face bounding box region to 112x112."""
        h, w, _ = frame.shape
        x, y, bw, bh = face_row[:4].astype(int)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        return cv2.resize(crop, target_size)

    @staticmethod
    def draw_feedback(
        frame: np.ndarray,
        faces: np.ndarray,
        status_text: str = "",
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draws bounding boxes, 5 facial landmarks, and status banner."""
        annotated = frame.copy()
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = face[-1]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            landmarks = face[4:14].reshape((5, 2)).astype(int)
            for (lx, ly) in landmarks:
                cv2.circle(annotated, (lx, ly), 2, (0, 255, 255), -1)

            cv2.putText(
                annotated,
                f"{confidence:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        if status_text:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                status_text,
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2
            )
        return annotated

    def release(self) -> None:
        """Safely releases hardware camera handle and closes OpenCV windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
