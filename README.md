# Computer Vision Learning Progress

This repository tracks my progress as I experiment with different computer vision models for various tasks.

## Completed

## In Progress
- Image classification


## To Learn
- Object detection
- Semantic segmentation
- Instance segmentation
- Pose estimation

- Image super-resolution
- Image denoising
- Image generation
- Image-to-image translation
- Style transfer
- 3D reconstruction

- Depth estimation
- Object tracking
- Optical flow

## Potential Datasets

- Welding Defect - Object Detection (https://www.kaggle.com/datasets/sukmaadhiwijaya/welding-defect-object-detection)
- Waymo (https://waymo.com/open/about/)
- Hard Hat Workers Dataset (https://public.roboflow.com/object-detection/hard-hat-workers/1)

## Project Ideas

- Rubik’s Cube Solver Assistant
    - Project idea: Use camera to detect cube faces, reconstruct cube state, and guide user step-by-step to solve (vision + algorithmic reasoning; similar systems use vision + planning models).
    - Tech stack: OpenCV + PyTorch (optional CNN for color detection) + classical CV (contours, perspective transform) + solver algorithm (Kociemba).
    - Dataset: Mostly self-collected images/video of cube states; optionally synthetic cube renders → shows geometry, CV fundamentals, and real-world robustness.
- Image → Anime Style Translation
    - Project idea: Convert real images (faces/scenes) into anime style using GAN-based image-to-image translation (CycleGAN / AnimeGAN).
    - Tech stack: PyTorch + GANs (CycleGAN / pix2pix) + training pipelines; CycleGAN works without paired datasets.
    - Dataset: Two unpaired sets: real images (e.g., ADE20K/photos) + anime images (frames from anime); CycleGAN learns mapping between domains.
- Multi-Object Tracking + Analytics: Real-Time Crowd Analytics System
    - Project idea: Build a real-time multi-object tracking + analytics system (track people, assign IDs, generate heatmaps & counts using ByteTrack, which links detections across frames including low-confidence ones for better tracking).
    - Tech stack: YOLO (detection) + ByteTrack (tracking) + OpenCV + PyTorch + FastAPI/Streamlit dashboard (real-time pipeline, tracking-by-detection system).
    - Dataset: Use MOT17/MOT20 + your own video footage; shows ability to work with real-world tracking benchmarks and custom data.
- Visual Search Engine
    - Project idea: Build a visual search engine where users upload an image → system returns visually similar images using deep features (modern systems extract embeddings from images and compare similarity, not exact matches ).
    - Tech stack: PyTorch + CLIP/ResNet (feature embeddings) + FAISS (fast nearest-neighbor search over vectors) + simple web UI (Streamlit/React); FAISS enables efficient similarity search at scale .
    - Dataset: DeepFashion / Google Landmarks / custom image collection; optionally mix image + text queries (CLIP allows both image-to-image and text-to-image retrieval in the same embedding space ).
- License Plate Recognition System
    - Project idea: Detect vehicles → extract license plates → OCR text.
    - Tech stack: YOLO (detection) + OCR (Tesseract/TrOCR) + OpenCV.
    - Dataset: OpenALPR / custom traffic footage; shows CV + NLP + real-world deployment.
- Plant Disease Detection (Agri AI)
    - Project idea: Upload plant image → classify disease + suggest treatment.
    - Tech stack: CNN / Vision Transformers + PyTorch + mobile/web app.
    - Dataset: PlantVillage; agriculture is a growing CV domain
- Industrial Defect Detection
    - Project idea: Detect defects in products (scratches, cracks, misprints).
    - Tech stack: CNN / segmentation models (U-Net) + PyTorch.
    - Dataset: MVTec AD; widely used in manufacturing CV systems
- Edge AI: Real-Time Worker Safety Monitor
    - Instead of a standard "people detector," build a system optimized for low-power hardware (like a Raspberry Pi or Jetson Nano) that monitors factory floor safety.
    - Project Idea: Detect if workers are wearing PPE (helmets, vests) and if they enter "no-go" zones near active machinery.
    - Tech Stack: YOLOv10/v11 (for speed) + TensorRT/OpenVINO (for edge optimization) + MQTT (for real-time alerts).
    - Dataset: Hard Hat Workers Dataset or self-curated frames.
    - Skills Earned: Model quantization, pruning, hardware-aware optimization, and real-time stream processing.
- Visual SLAM for Indoor Navigation
    - Project Idea: Use a smartphone camera to map a room in 3D and calculate the "ego-motion" (path) of the device as it moves through the space.
    - Tech Stack: ORB-SLAM3 or OpenCV (SFM module) + Python/C++.
    - Dataset: TUM RGB-D or KITTI.
    - Skills Earned: 3D Geometry, Camera Calibration, Point Cloud processing, and Coordinate Transformations.
- Vision-Based Drone Navigation (Robotics & Defense)
    - Project Idea: Build a "Follow-Me" drone algorithm that uses Monocular Depth Estimation to avoid obstacles while tracking a moving target (e.g., a hiker or a vehicle).
    - Tech Stack: MiDaS or ZoeDepth (Depth) + Purdue's Spiking Neural Networks (SNNs) or standard PyTorch + ROS2 (Robot Operating System).
    - Dataset: AODRaw (helps with navigation in fog/rain) or TUM RGB-D.
    - Skills Earned: 3D scene reconstruction, ROS2 integration, and handling "Adverse Weather" domain gaps.
- "Zero-Shot" Anomaly Detection for Manufacturing
    - Most manufacturing lines don't have thousands of images of "broken" parts. You need to train a model on what a "good" part looks like and let it figure out what "bad" looks like on its own.
    - Project Idea: A "Universal Quality Inspector." Train a model only on "perfect" circuit boards. When a board with a tiny scratch or missing solder point passes by, the model flags it as an anomaly.
    - Tech Stack: PatchCore or Anomalib (Intel's library) + PyTorch + Gradio (for the UI).
    - Dataset: MVTec AD is the industry gold standard here.