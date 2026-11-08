# Anti-Spoofing Facial Recognition Authentication System (Phase 1)

> **Graduation Project — Cybersecurity-Oriented Biometric Authentication**  
> **Project Title:** Design and Implementation of an Anti-Spoofing Facial Recognition Authentication System Using Liveness Detection  
> **Student Team:** Al-Haitham Ahmed Al-Dabbi, Adham Atef Abood Al-Hagri, Majd Al-deen Ahmed Amer  
> **Supervisor:** Saleh Al-Rayashi, Raghad Al-Qairi
> **Academic Year:** 2026  

---

## 📌 Executive Summary & Overview (Week 1: Software Foundation)

This repository contains the **Phase 1 (Days 1–7)** core software foundation for an Anti-Spoofing Facial Recognition Authentication System. The foundation was designed, implemented, and benchmarked on a 64-bit development environment (AMD Ryzen 7 7735HS, NVIDIA GeForce RTX 4060, Windows 11).

Instead of disconnected scripts, the architecture provides a modular, production-ready pipeline covering real-time video capture, Deep Neural Network (DNN) face detection, automated image quality gatekeeping (lighting and sharpness verification), privacy-aware 5-angle user enrollment with explicit consent, and 128-dimensional embedding identity verification.

### Key Features Implemented in Phase 1
- **Fast DNN Face & Landmark Detection:** Uses OpenCV's lightweight **YuNet (`face_detection_yunet_2023mar.onnx`)** model to detect faces and 5 facial landmarks in real time.
- **Automated Quality Gatekeeper:** Evaluates mean grayscale illumination ($\mu_{\text{brightness}}$) and Laplacian variance sharpness ($\sigma^2_{\text{Laplacian}}$) on $112 \times 112$ standardized face crops, blocking blurry, overexposed, or under-illuminated frames.
- **Single-Face Security Policy:** Automatically blocks enrollment and authentication if zero faces or multiple faces ($>1$) appear in the camera frame.
- **Privacy-Preserving Enrollment:** Enforces explicit user consent prior to capture, generates non-reversible deterministic SHA-256 derived user IDs (`USR_<12-HEX>`), and stores normalized 128-D embeddings rather than raw biometric images.
- **ArcFace/SFace Verification Engine:** Extracts 128-D feature vectors via **SFace (`face_recognition_sface_2021dec.onnx`)** and verifies identities using a weighted centroid + multi-angle cosine similarity score ($\tau = 0.42$).
- **One-Key CSV Evaluation Logger:** Built-in benchmarking tool that logs cosine similarity scores, lighting metrics, sharpness, and frame latency (ms) directly to `logs/week1_evaluation_results.csv`.

---

## 📂 Project Directory Structure

```text
C:\anti_spoofing_system\
│
├── .venv\                                      # Isolated Python 3.14 virtual environment
├── models\                                     # Pre-trained ONNX Deep Learning models
│   ├── face_detection_yunet_2023mar.onnx       # YuNet Face & 5-Landmark Detector (232 KB)
│   └── face_recognition_sface_2021dec.onnx     # SFace 128-D Embedding Model (38.7 MB)
│
├── src\                                        # Core modular Python source package
│   ├── __init__.py                             # Package initializer
│   ├── camera_processor.py                     # Days 1-3: Capture, Detection & Quality module
│   └── recognition_engine.py                   # Days 4-5: Enrollment & Cosine Matching module
│
├── data\
│   └── enrolled_users\                         # Structured biometric template repository
│       └── USR_5364F2F2FC4F\                   # Derived ID folder for enrolled user
│           └── template.json                   # Centroid & 5-angle 128-D embeddings
│
├── logs\
│   └── week1_evaluation_results.csv            # Day 6: Quantitative evaluation test log
│
├── tests\                                      # Reserved for unit and security test scripts
├── main.py                                     # Day 7: Main interactive application & CLI
└── README.md                                   # Complete project technical documentation & setup guide
```

---

## ⚙️ Installation & Setup (Step-by-Step)

### Prerequisites
- **Operating System:** Windows 10/11 (64-bit) or Linux / Raspberry Pi OS
- **Python:** Python 3.10+ (Tested on Python 3.14 64-bit)
- **Hardware:** Integrated Webcam or USB RGB Camera

### 1. Create the Project Directory & Folders
Open **Windows PowerShell** and run:

```powershell
mkdir C:\anti_spoofing_system
cd C:\anti_spoofing_system
mkdir src, data, data\enrolled_users, models, logs, tests
```

### 2. Create & Activate the Virtual Environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```
*(Verify that your terminal prompt begins with `(.venv) PS C:\anti_spoofing_system>` before proceeding).*

### 3. Install Required Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy onnxruntime cryptography
```

### 4. Download Pre-Trained ONNX Models
Download the official OpenCV Zoo **YuNet** and **SFace** models into the `models\` directory:

```powershell
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -OutFile "models\face_detection_yunet_2023mar.onnx"
Invoke-WebRequest -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -OutFile "models\face_recognition_sface_2021dec.onnx"
```

---

## 🚀 How to Run & Use the System

Whenever you open a new PowerShell window, activate the environment and launch `main.py`:

```powershell
cd C:\anti_spoofing_system
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
python main.py
```

### Interactive CLI Menu

```text
========================================================
 WEEK 1: ANTI-SPOOFING FACIAL AUTHENTICATION FOUNDATION
========================================================
 Enrolled Users Currently Loaded: 1
 1. [Days 1-3] Run Camera, Face Detection & Quality Test
 2. [Day 4]    Register / Enroll a New User (5 Angles)
 3. [Days 5-6] Run Face Recognition & Log Test Results
 4. [Exit]     Quit Application
```

#### Option `1`: Camera, Face Detection & Quality Test (Days 1–3)
- Opens the live webcam feed with bounding boxes, 5 facial landmarks, and a $112 \times 112$ standardized face preview in the top-right corner.
- Displays real-time **Brightness** and **Sharpness** scores:
  - **Green Box (`Quality OK`):** Brightness between $45$ and $220$, Sharpness $\ge 40$.
  - **Orange Box (`Too Dark` / `Too Bright` / `Blurry`):** Alerts user to adjust lighting or hold still.
  - **Red/Orange Alert (`No Face` / `Multiple Faces`):** Enforces single-user framing policy.
- **Controls:** Click inside the video window and press **`q`** to return to the menu.

#### Option `2`: Guided 5-Angle User Registration (Day 4)
- Prompts for a **username** and requires explicit **`yes`/`no` biometric consent** (typing `no` aborts enrollment immediately).
- Guides the user through **5 multi-angle captures**:
  1. Sample 1/5: Look straight at the camera
  2. Sample 2/5: Turn head slightly LEFT
  3. Sample 3/5: Turn head slightly RIGHT
  4. Sample 4/5: Tilt head slightly UP
  5. Sample 5/5: Normal expression, look straight
- **Controls:** Press **`SPACEBAR`** when the bounding box is **GREEN** to capture each sample, or **`q`** to cancel. (The system blocks capture if quality conditions are not met).
- Computes the normalized 128-D centroid vector and saves the profile to `data/enrolled_users/USR_<HASH>/template.json`.

#### Option `3`: Real-Time Face Recognition & Evaluation Logger (Days 5–6)
- Prompts for a **Test Condition** label (e.g., `Normal_Light`, `Low_Light`, `Angle_30deg`, `Unregistered`) and the **Expected Identity** (e.g., `solo` or `Unknown`).
- Performs real-time identity verification:
  - **Green Box:** Matched enrolled user ($S_{\text{combined}} \ge 0.42$).
  - **Red Box:** Unregistered / Unknown user ($S_{\text{combined}} < 0.42$).
- **Controls:**
  - Press **`l`** inside the video window to log current trial metrics (Timestamp, Condition, Expected ID, Predicted ID, Cosine Score, Brightness, Sharpness, Latency in ms, and `PASS`/`FAIL` outcome) to `logs/week1_evaluation_results.csv`.
  - Press **`q`** to return to the menu.

---

## 🧮 Technical & Mathematical Background

### 1. Image Quality Metrics (`src/camera_processor.py`)
Before any face crop is enrolled or evaluated, the extracted face region $I_{\text{gray}}$ must pass two automated checks:
* **Illumination (Mean Brightness):**
  $$\mu_{\text{brightness}} = \frac{1}{H \times W} \sum_{x=1}^{W} \sum_{y=1}^{H} I_{\text{gray}}(x, y)$$
  *Accepted Range:* $45.0 \le \mu_{\text{brightness}} \le 220.0$.
* **Focus / Sharpness (Variance of the Laplacian):**
  $$\sigma^2_{\text{Laplacian}} = \text{Var}(\nabla^2 I_{\text{gray}}) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
  *Accepted Threshold:* $\sigma^2_{\text{Laplacian}} \ge 40.0$.

### 2. Privacy-Aware Non-Sensitive User ID (`src/recognition_engine.py`)
To ensure privacy compliance, raw usernames are never stored directly as storage keys. A deterministic cryptographic SHA-256 hash generates an anonymized identifier:
$$\text{user\_id} = \text{"USR\_"} \parallel \text{Upper}\Big(\text{SHA256}\big(\text{Lower}(\text{username})\big)_{[0:12]}\Big)$$
*Example:* Username `"solo"` maps to `USR_5364F2F2FC4F`.

### 3. Cosine Similarity Verification Engine (`src/recognition_engine.py`)
Each aligned $112 \times 112$ face produces an $\ell_2$-normalized 128-dimensional embedding vector $\mathbf{e} \in \mathbb{R}^{128}$ where $\|\mathbf{e}\|_2 = 1$. Verification compares the live vector $\mathbf{e}_{\text{live}}$ against the template centroid $\mathbf{e}_{\text{centroid}}$ and all sample vectors $\{\mathbf{s}_1, \dots, \mathbf{s}_5\}$:
$$S_{\text{centroid}} = \mathbf{e}_{\text{live}} \cdot \mathbf{e}_{\text{centroid}}, \quad S_{\text{max\_sample}} = \max_{i \in \{1..5\}} (\mathbf{e}_{\text{live}} \cdot \mathbf{s}_i)$$
$$S_{\text{combined}} = 0.6 \cdot S_{\text{centroid}} + 0.4 \cdot S_{\text{max\_sample}}$$
Access is granted if and only if $S_{\text{combined}} \ge \tau$ (where $\tau = 0.42$).

---

## 📊 Summary of Configured Parameters

| Parameter / Metric | Formula / Source | Configured Threshold | Objective |
| :--- | :--- | :--- | :--- |
| **Face Detection** | YuNet DNN (`FaceDetectorYN`) | $\text{Confidence} \ge 0.85$ | Real-time face location & 5 facial landmarks |
| **Brightness Check** | $\mu = \frac{1}{HW}\sum I_{\text{gray}}(x,y)$ | $45.0 \le \mu \le 220.0$ | Rejects low-light and overexposed frames |
| **Sharpness Check** | $\sigma^2 = \text{Var}(\nabla^2 I_{\text{gray}})$ | $\sigma^2 \ge 40.0$ | Rejects motion-blurred or out-of-focus frames |
| **Derived User ID** | `"USR_" + \text{SHA256}(\text{username})_{[0:12]}$ | 12-character Hex String | Anonymizes user profile storage directory |
| **Cosine Similarity** | $S = 0.6 S_{\text{centroid}} + 0.4 S_{\text{max\_sample}}$ | $\tau = 0.42$ | Decision boundary for identity verification |

---

## ✅ Deliverable Traceability Matrix & Test Results

| Day | Assigned Task | Status | Implementation & Verification Summary |
| :--- | :--- | :--- | :--- |
| **Day 1** | Virtual environment setup, dependencies, folder layout, webcam test | **Completed** | Created `.venv`; fetched YuNet & SFace ONNX models; verified camera capture. |
| **Day 2** | Real-time frame capture, YuNet face/landmark detection, safe exit | **Completed** | Built `src/camera_processor.py` with bounding box overlays and hardware cleanup. |
| **Day 3** | Face extraction ($112 \times 112$), quality control, multi-face enforcement | **Completed** | Implemented mean brightness and Laplacian variance checks; enforced single-face policy. |
| **Day 4** | Guided enrollment UI, consent check, 5-angle capture, JSON storage | **Completed** | Added enrollment flow in `main.py` & `src/recognition_engine.py`. Verified consent abort. |
| **Day 5** | Embedding extraction via SFace, cosine matching algorithm | **Completed** | Integrated SFace model and weighted cosine metric ($\tau = 0.42$) decision logic. |
| **Day 6** | Systematic evaluation, multi-condition testing, CSV logging | **Completed** | Tested registered/unregistered subjects under varied conditions; logged to `logs/week1_evaluation_results.csv`. |
| **Day 7** | Refactoring, modular code organization, documentation finalization | **Completed** | Modularized source packages, finalized `main.py` CLI dashboard, and unified documentation. |

### Verified Test Log Sample (Day 6 Telemetry)

* **Consent Verification Test:**
  * Input: `username = solo`, `consent = no`
  * System Output: `[ABORTED] Explicit user consent is required.`
* **Enrollment Verification Test:**
  * Input: `username = solo`, `consent = yes`, `5 samples captured`
  * System Output: `[SUCCESS] User 'solo' registered with Derived ID: USR_5364F2F2FC4F`
* **Recognition Evaluation Sample (`logs/week1_evaluation_results.csv`):**
  * Trial 1 (`Normal_Light`): Expected=`solo` | Predicted=`solo` | Cosine Score=`0.769` | Outcome=`PASS`