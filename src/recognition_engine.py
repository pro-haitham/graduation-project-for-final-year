import os
import json
import hashlib
from datetime import datetime, timezone
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

class RecognitionEngine:
    """
    Manages SFace (ArcFace-based) 128-D face embedding extraction,
    user enrollment storage, and cosine similarity identity verification.
    """
    def __init__(
        self,
        recognizer_model_path: str = "models/face_recognition_sface_2021dec.onnx",
        db_dir: str = "data/enrolled_users",
        cosine_threshold: float = 0.42
    ) -> None:
        if not os.path.exists(recognizer_model_path):
            raise FileNotFoundError(
                f"Recognizer model missing at '{recognizer_model_path}'."
            )

        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognizer_model_path,
            config=""
        )
        self.db_dir = db_dir
        self.cosine_threshold = cosine_threshold
        os.makedirs(self.db_dir, exist_ok=True)
        self.enrolled_templates: Dict[str, Dict[str, Any]] = self._load_all_templates()

    def extract_embedding(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Aligns the detected face using its 5 landmarks and extracts a normalized 128-D embedding."""
        aligned_face = self.recognizer.alignCrop(frame, face_row)
        embedding = self.recognizer.feature(aligned_face)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @staticmethod
    def _derive_user_id(username: str) -> str:
        """Creates a non-sensitive derived identifier."""
        clean_name = username.strip().lower().encode("utf-8")
        return "USR_" + hashlib.sha256(clean_name).hexdigest()[:12].upper()

    def register_user(self, username: str, embeddings: List[np.ndarray], consent_given: bool = True) -> str:
        """Saves averaged + individual multi-angle embeddings for a consenting user."""
        if not consent_given:
            raise PermissionError("User registration aborted: explicit biometric consent is required.")
        if not username or not username.strip():
            raise ValueError("Username cannot be empty.")
        if len(embeddings) == 0:
            raise ValueError("At least one valid face embedding is required for registration.")

        user_id = self._derive_user_id(username)
        user_folder = os.path.join(self.db_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)

        stacked = np.vstack(embeddings)
        centroid = np.mean(stacked, axis=0, keepdims=True)
        centroid = centroid / np.linalg.norm(centroid)

        record = {
            "user_id": user_id,
            "username": username.strip(),
            "consent": consent_given,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(embeddings),
            "centroid_embedding": centroid.flatten().tolist(),
            "sample_embeddings": [emb.flatten().tolist() for emb in embeddings]
        }

        template_path = os.path.join(user_folder, "template.json")
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        self.enrolled_templates[user_id] = record
        return user_id

    def _load_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Loads all enrolled user templates from disk into memory."""
        templates: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(self.db_dir):
            return templates

        for entry in os.listdir(self.db_dir):
            template_file = os.path.join(self.db_dir, entry, "template.json")
            if os.path.isfile(template_file):
                try:
                    with open(template_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        templates[data["user_id"]] = data
                except (json.JSONDecodeError, KeyError):
                    continue
        return templates

    def verify_identity(self, live_embedding: np.ndarray) -> Tuple[bool, str, float]:
        """Compares a live face embedding against all stored templates."""
        if not self.enrolled_templates:
            return False, "No Enrolled Users", 0.0

        best_score = -1.0
        best_username = "Unknown"
        live_vec = live_embedding.flatten()

        for user_id, data in self.enrolled_templates.items():
            centroid_vec = np.array(data["centroid_embedding"], dtype=np.float32)
            centroid_score = float(np.dot(live_vec, centroid_vec))

            sample_scores = [
                float(np.dot(live_vec, np.array(s, dtype=np.float32)))
                for s in data.get("sample_embeddings", [])
            ]
            max_sample_score = max(sample_scores) if sample_scores else centroid_score
            combined_score = 0.6 * centroid_score + 0.4 * max_sample_score

            if combined_score > best_score:
                best_score = combined_score
                best_username = data["username"]

        if best_score >= self.cosine_threshold:
            return True, best_username, best_score
        return False, "Unknown", best_score
