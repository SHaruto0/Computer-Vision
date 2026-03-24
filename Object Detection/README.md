OBJECT DETECTION ROADMAP (START → END)

Stage 1 — Foundations

Implement YOLOv1 from scratch
Implement IoU (Intersection over Union)
Implement Non-Max Suppression (NMS)
Train on Pascal VOC or Oxford-IIIT Pet Dataset

Stage 2 — Core Detection Concepts

Implement SSD (Single Shot Detector)
Learn and implement anchor boxes
Learn ground truth ↔ anchor matching
(Optional) Implement focal loss (RetinaNet concept)

Stage 3 — Modern One-Stage Detection

Implement YOLOv3 (or simplified YOLOv5-style)
Learn multi-scale prediction
Learn Feature Pyramid Networks (FPN)
Train on MS COCO (or subset)

Stage 4 — Two-Stage Detectors

Implement Faster R-CNN (simplified)
Learn Region Proposal Networks (RPN)
Learn ROI Pooling / ROI Align
Understand multi-stage pipelines

Stage 5 — Transformer-Based Detection

Implement DETR (simplified)
Learn set prediction (no anchors)
Learn Hungarian matching loss
Learn attention mechanisms

Stage 6 — Specialization (choose one)

Path A — Autonomous Driving

Learn KITTI Dataset or Waymo Open Dataset
Implement PointPillars or CenterPoint
Learn 3D object detection

Path B — Real-Time / Industry

Optimize YOLO-style models
Learn ONNX / TensorRT deployment
Focus on latency vs accuracy tradeoffs

Path C — Research / Advanced Models

Study Deformable DETR, DINO
Explore multi-modal transformers
Work on new architectures