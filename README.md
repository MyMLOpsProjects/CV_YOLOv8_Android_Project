# YOLOv8 Object Detection & Segmentation for Android

A high-performance, real-time computer vision application for Android leveraging **YOLOv8** and **TensorFlow Lite (LiteRT)**. This project supports both standard object detection and **Instance Segmentation**, optimized for mobile hardware.

## 🚀 Key Features

- **Instance Segmentation**: Real-time mask generation using YOLOv8-seg models.
- **Real-time Performance**: Optimized pipeline reaching **< 30ms** inference on modern devices.
- **Hardware Acceleration**: Full support for Android **GPU Delegate** with safe CPU fallback.
- **INT8 Quantization**: Support for fully quantized models (UINT8/INT8) for maximum efficiency and reduced model size (~3.5MB).
- **Spatial Mask Cropping**: High-accuracy segmentation alignment through intelligent mask-to-box cropping.
- **Modern Camera Stack**: Built with **Jetpack CameraX** supporting full-screen 16:9 previews and optimized image analysis.
- **Advanced Preprocessing**: High-speed manual transposition (HWC to CHW) implemented with direct array access to eliminate overhead.

## 📊 Performance Benchmark

| Model | Type | Acceleration | Avg. Inference |
| :--- | :--- | :--- | :--- |
| YOLOv8n (Float32) | Detection | CPU (4 threads) | ~150ms |
| YOLOv8n (Float32) | Detection | **GPU Delegate** | **~40ms** |
| YOLOv8n (INT8) | Detection | **GPU/NPU** | **~15ms** |
| YOLOv8n-seg (INT8)| **Segmentation**| **GPU/NPU** | **~25ms** |

*Note: Benchmarks performed on mid-to-high range Android devices.*

## 🛠️ Setup & Installation

### Prerequisites
- Android Studio Iguana or newer.
- Android device with API 24+ and USB Debugging enabled.

### Steps
1. **Clone the project**:
   ```bash
   git clone <your-repo-url>
   ```
2. **Configure Assets**:
   - Place your model (e.g., `yolov8n-seg_int8.tflite`) and `labels.txt` in `app/src/main/assets`.
   - Update the model filename in `MainActivity.kt`.
3. **Build**:
   - Open in Android Studio, sync Gradle, and click **Run**.

## 🏗️ Technical Architecture

- **Engine**: TensorFlow Lite Interpreter
- **Language**: 100% Idiomatic Kotlin
- **Memory**: Optimized direct `ByteBuffer` reuse to minimize GC pressure.
- **UI**: Custom hardware-accelerated `OverlayView` for bounding boxes and semi-transparent segmentation masks.

## 📅 Roadmap
- [x] Migrate from ONNX to TFLite.
- [x] Enable GPU hardware acceleration.
- [x] Implement Instance Segmentation (YOLOv8-seg).
- [x] Support INT8 Quantized models.
- [ ] Integrate ByteTrack for persistent object tracking.
- [ ] Support for dynamic model switching from UI.

---
*Developed for high-performance mobile AI development.*
