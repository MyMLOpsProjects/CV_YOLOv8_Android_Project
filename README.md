# YOLOv8 Object Detection for Android

A real-time object detection application for Android using the **YOLOv8 Nano** model and **TensorFlow Lite (LiteRT)**. This project is highly optimized for mobile performance, leveraging hardware acceleration and efficient data processing.

## 🚀 Key Features

- **Real-time Detection**: Optimized pipeline for low-latency inference.
- **TFLite Integration**: Uses TensorFlow Lite for native mobile machine learning performance.
- **GPU Acceleration**: Utilizes the Android GPU Delegate for fast mathematical computations on supported devices.
- **Full-Screen Camera**: Implemented with Jetpack CameraX for a smooth, modern 16:9 camera preview experience.
- **Dynamic Bounding Boxes**: Custom `OverlayView` for high-contrast green bounding boxes and legible object labels.
- **Optimized Preprocessing**: Custom manual transposition logic (HWC to CHW) optimized for YOLOv8's expected input format.

## 📊 Performance

| Model | Acceleration | Average Inference Time |
| :--- | :--- | :--- |
| YOLOv8n (Float32) | CPU (4 threads) | ~150ms - 200ms |
| YOLOv8n (Float32) | **GPU Delegate** | **< 50ms** |

*Note: Performance results based on modern Android hardware.*

## 🛠️ Setup & Installation

### Prerequisites
- Android Studio Iguana or newer.
- An Android device (API 24+) with USB Debugging enabled.

### Steps
1. **Clone the project**:
   ```bash
   git clone <your-repo-url>
   ```
2. **Add Model Assets**:
   - Ensure `yolov8n.tflite` and `labels.txt` are present in `app/src/main/assets`.
3. **Build and Run**:
   - Open the project in Android Studio.
   - Sync Gradle.
   - Click **Run 'app'**.

## 🏗️ Technical Architecture

- **Language**: 100% Kotlin
- **ML Framework**: TensorFlow Lite (org.tensorflow:tensorflow-lite)
- **Camera API**: Jetpack CameraX
- **Optimization**: Sequential array transposition for planar data handling.

## 📅 Roadmap
- [x] Migrate from ONNX to TFLite.
- [x] Enable GPU hardware acceleration.
- [ ] Implement INT8 Quantized model support for further speedup.
- [ ] Add support for custom model loading from external storage.

---
*Developed for high-performance mobile AI development.*
