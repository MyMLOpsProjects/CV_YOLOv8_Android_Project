package com.example.yolo

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

class YoloDetector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener,
) {

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 640
    private var tensorHeight = 640
    private var numChannel = 0
    private var numElements = 0

    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null
    private var hwcArray: FloatArray? = null
    private var chwArray: FloatArray? = null
    private var outputArray: FloatArray? = null

    private var imageProcessor: ImageProcessor? = null

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                Log.d("YoloDetector", "GPU acceleration enabled")
            } else {
                options.setNumThreads(4)
                Log.d("YoloDetector", "GPU not supported, using CPU")
            }
        } catch (e: Exception) {
            Log.e("YoloDetector", "Exception initializing GPU delegate", e)
            options.setNumThreads(4)
        } catch (e: Error) {
            Log.e("YoloDetector", "Error initializing GPU delegate (Class not found?)", e)
            options.setNumThreads(4)
        }
        
        val interp = Interpreter(model, options)
        interpreter = interp

        val inputShape = interp.getInputTensor(0).shape()
        val outputShape = interp.getOutputTensor(0).shape()

        if (inputShape[1] == 3) {
            // NCHW [1, 3, 640, 640]
            tensorWidth = inputShape[3]
            tensorHeight = inputShape[2]
            val totalSize = 3 * tensorHeight * tensorWidth
            inputBuffer = ByteBuffer.allocateDirect(totalSize * 4).order(ByteOrder.nativeOrder())
            hwcArray = FloatArray(totalSize)
            chwArray = FloatArray(totalSize)
        } else {
            // NHWC [1, 640, 640, 3]
            tensorWidth = inputShape[2]
            tensorHeight = inputShape[1]
        }

        numChannel = outputShape[1]
        numElements = outputShape[2]
        
        outputBuffer = ByteBuffer.allocateDirect(1 * numChannel * numElements * 4).order(ByteOrder.nativeOrder())
        outputArray = FloatArray(numChannel * numElements)

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(tensorHeight, tensorWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()

        val inputStream = context.assets.open(labelPath)
        val reader = BufferedReader(InputStreamReader(inputStream))
        var line: String? = reader.readLine()
        while (line != null) {
            labels.add(line.trim())
            line = reader.readLine()
        }
        reader.close()
        inputStream.close()
    }

    fun detect(frame: Bitmap) {
        val interp = interpreter ?: return
        val outBuffer = outputBuffer ?: return
        val processor = imageProcessor ?: return
        val outArray = outputArray ?: return

        val startTime = SystemClock.uptimeMillis()

        // 1. Preprocessing
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(frame)
        val processedImage = processor.process(tensorImage)

        val runInput = if (inputBuffer != null) {
            val srcBuffer = processedImage.buffer.asFloatBuffer()
            srcBuffer.rewind()
            srcBuffer.get(hwcArray!!)
            
            val hwc = hwcArray!!
            val chw = chwArray!!
            val imgSize = tensorHeight * tensorWidth
            
            // Highly optimized sequential transposition
            var hwcIdx = 0
            for (i in 0 until imgSize) {
                chw[i] = hwc[hwcIdx++]
                chw[imgSize + i] = hwc[hwcIdx++]
                chw[2 * imgSize + i] = hwc[hwcIdx++]
            }
            
            inputBuffer!!.rewind()
            inputBuffer!!.asFloatBuffer().put(chw)
            inputBuffer!!
        } else {
            processedImage.buffer
        }

        // 2. Inference
        outBuffer.rewind()
        interp.run(runInput, outBuffer)

        // 3. Postprocessing
        outBuffer.rewind()
        outBuffer.asFloatBuffer().get(outArray)

        val bestBoxes = bestBox(outArray)
        val totalTime = SystemClock.uptimeMillis() - startTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, totalTime)
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            
            var classOffset = c + (numElements * 4)
            for (j in 4 until numChannel) {
                val conf = array[classOffset]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = j - 4
                }
                classOffset += numElements
            }

            if (maxIdx != -1) {
                var cx = array[c]
                var cy = array[c + numElements]
                var w = array[c + (numElements * 2)]
                var h = array[c + (numElements * 3)]

                // YOLOv8 TFLite outputs are often in pixels (0..640)
                if (cx > 2f || w > 2f) {
                    cx /= tensorWidth
                    cy /= tensorHeight
                    w /= tensorWidth
                    h /= tensorHeight
                }

                val x1 = cx - (w / 2f)
                val y1 = cy - (h / 2f)
                val x2 = cx + (w / 2f)
                val y2 = cy + (h / 2f)

                if (x1 < -0.1f || x1 > 1.1f) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = labels.getOrNull(maxIdx) ?: "unknown"
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (calculateIoU(first, next) > IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)

        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.35f
        private const val IOU_THRESHOLD = 0.45f
    }
}
