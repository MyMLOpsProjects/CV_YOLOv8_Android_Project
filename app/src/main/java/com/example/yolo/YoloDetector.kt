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
    
    private var isSegmentation = false
    private var numMasks = 32
    private var maskWidth = 160
    private var maskHeight = 160

    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer0: ByteBuffer? = null
    private var outputBuffer1: ByteBuffer? = null
    
    private var hwcByteArray: ByteArray? = null
    private var chwByteArray: ByteArray? = null
    private var hwcArray: FloatArray? = null
    private var chwArray: FloatArray? = null
    
    private var outputArray0: FloatArray? = null
    private var outputArray1: FloatArray? = null

    private var imageProcessor: ImageProcessor? = null
    private var isQuantized = false

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        
        try {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate()
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
            Log.e("YoloDetector", "Error initializing GPU delegate", e)
            options.setNumThreads(4)
        }
        
        val interp = Interpreter(model, options)
        interpreter = interp

        val inputTensor = interp.getInputTensor(0)
        val inputShape = inputTensor.shape()
        isQuantized = inputTensor.dataType() == DataType.UINT8 || inputTensor.dataType() == DataType.INT8

        // Detect if it's a segmentation model by checking number of outputs
        isSegmentation = interp.outputTensorCount > 1

        if (inputShape[1] == 3) {
            tensorWidth = inputShape[3]
            tensorHeight = inputShape[2]
            val totalSize = 3 * tensorHeight * tensorWidth
            val elementSize = if (isQuantized) 1 else 4
            inputBuffer = ByteBuffer.allocateDirect(totalSize * elementSize).order(ByteOrder.nativeOrder())
            if (isQuantized) {
                hwcByteArray = ByteArray(totalSize)
                chwByteArray = ByteArray(totalSize)
            } else {
                hwcArray = FloatArray(totalSize)
                chwArray = FloatArray(totalSize)
            }
        } else {
            tensorWidth = inputShape[2]
            tensorHeight = inputShape[1]
        }

        // Output 0: Detections [1, 4+num_classes+32, 8400]
        val outputTensor0 = interp.getOutputTensor(0)
        val outputShape0 = outputTensor0.shape()
        numChannel = outputShape0[1]
        numElements = outputShape0[2]
        outputBuffer0 = ByteBuffer.allocateDirect(numChannel * numElements * 4).order(ByteOrder.nativeOrder())
        outputArray0 = FloatArray(numChannel * numElements)

        if (isSegmentation) {
            // Output 1: Proto masks [1, 32, 160, 160]
            val outputTensor1 = interp.getOutputTensor(1)
            val outputShape1 = outputTensor1.shape()
            numMasks = outputShape1[1]
            maskHeight = outputShape1[2]
            maskWidth = outputShape1[3]
            outputBuffer1 = ByteBuffer.allocateDirect(numMasks * maskHeight * maskWidth * 4).order(ByteOrder.nativeOrder())
            outputArray1 = FloatArray(numMasks * maskHeight * maskWidth)
        }

        val processorBuilder = ImageProcessor.Builder()
            .add(ResizeOp(tensorHeight, tensorWidth, ResizeOp.ResizeMethod.BILINEAR))
        if (isQuantized) {
            processorBuilder.add(CastOp(DataType.UINT8))
        } else {
            processorBuilder.add(NormalizeOp(0f, 255f))
            processorBuilder.add(CastOp(DataType.FLOAT32))
        }
        imageProcessor = processorBuilder.build()

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
        val processor = imageProcessor ?: return
        val outBuf0 = outputBuffer0 ?: return
        val outArr0 = outputArray0 ?: return

        val startTime = SystemClock.uptimeMillis()

        val tensorImage = TensorImage(if (isQuantized) DataType.UINT8 else DataType.FLOAT32)
        tensorImage.load(frame)
        val processedImage = processor.process(tensorImage)

        val runInput = if (inputBuffer != null) {
            if (isQuantized) {
                val byteBuffer = processedImage.buffer
                byteBuffer.rewind()
                byteBuffer.get(hwcByteArray!!)
                val hwc = hwcByteArray!!
                val chw = chwByteArray!!
                val imgSize = tensorHeight * tensorWidth
                var hwcIdx = 0
                for (i in 0 until imgSize) {
                    chw[i] = hwc[hwcIdx++]
                    chw[imgSize + i] = hwc[hwcIdx++]
                    chw[2 * imgSize + i] = hwc[hwcIdx++]
                }
                inputBuffer!!.rewind()
                inputBuffer!!.put(chw)
                inputBuffer!!
            } else {
                val floatBuffer = processedImage.buffer.asFloatBuffer()
                floatBuffer.rewind()
                floatBuffer.get(hwcArray!!)
                val hwc = hwcArray!!
                val chw = chwArray!!
                val imgSize = tensorHeight * tensorWidth
                var hwcIdx = 0
                for (i in 0 until imgSize) {
                    chw[i] = hwc[hwcIdx++]
                    chw[imgSize + i] = hwc[hwcIdx++]
                    chw[2 * imgSize + i] = hwc[hwcIdx++]
                }
                inputBuffer!!.rewind()
                inputBuffer!!.asFloatBuffer().put(chw)
                inputBuffer!!
            }
        } else {
            processedImage.buffer
        }

        val outputs = mutableMapOf<Int, Any>(0 to outBuf0)
        if (isSegmentation) {
            outputs[1] = outputBuffer1!!
        }

        outBuf0.rewind()
        outputBuffer1?.rewind()
        interp.runForMultipleInputsOutputs(arrayOf(runInput), outputs)

        outBuf0.rewind()
        outBuf0.asFloatBuffer().get(outArr0)
        
        if (isSegmentation) {
            outputBuffer1!!.rewind()
            outputBuffer1!!.asFloatBuffer().get(outputArray1!!)
        }

        val bestBoxes = bestBox(outArr0)
        val totalTime = SystemClock.uptimeMillis() - startTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, totalTime)
    }

    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()
        val numClasses = labels.size

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            
            for (j in 0 until numClasses) {
                val conf = array[c + (numElements * (j + 4))]
                if (conf > maxConf) {
                    maxConf = conf
                    maxIdx = j
                }
            }

            if (maxIdx != -1) {
                var cx = array[c]
                var cy = array[c + numElements]
                var w = array[c + (numElements * 2)]
                var h = array[c + (numElements * 3)]

                // Always normalize coordinates
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

                if (x1 < -0.1f || x1 > 1.1f || y1 < -0.1f || y1 > 1.1f) continue

                var maskCoeffs: FloatArray? = null
                if (isSegmentation) {
                    maskCoeffs = FloatArray(numMasks)
                    for (i in 0 until numMasks) {
                        maskCoeffs[i] = array[c + (numElements * (numClasses + 4 + i))]
                    }
                }

                val maskData = if (isSegmentation) processMask(maskCoeffs!!, x1, y1, x2, y2) else null

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = labels.getOrNull(maxIdx) ?: "unknown",
                        mask = maskData?.first,
                        mWidth = maskData?.second ?: 0,
                        mHeight = maskData?.third ?: 0
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun processMask(coeffs: FloatArray, x1: Float, y1: Float, x2: Float, y2: Float): Triple<FloatArray, Int, Int> {
        val proto = outputArray1!!
        val maskSize = maskHeight * maskWidth
        
        val mx1 = (x1 * maskWidth).toInt().coerceIn(0, maskWidth - 1)
        val my1 = (y1 * maskHeight).toInt().coerceIn(0, maskHeight - 1)
        val mx2 = (x2 * maskWidth).toInt().coerceIn(0, maskWidth - 1)
        val my2 = (y2 * maskHeight).toInt().coerceIn(0, maskHeight - 1)
        
        val cropWidth = mx2 - mx1 + 1
        val cropHeight = my2 - my1 + 1
        
        if (cropWidth <= 0 || cropHeight <= 0) return Triple(FloatArray(0), 0, 0)
        
        val result = FloatArray(cropWidth * cropHeight)
        
        for (y in 0 until cropHeight) {
            val maskY = my1 + y
            for (x in 0 until cropWidth) {
                val maskX = mx1 + x
                val protoIdx = maskY * maskWidth + maskX
                
                var sum = 0f
                for (j in 0 until numMasks) {
                    sum += coeffs[j] * proto[j * maskSize + protoIdx]
                }
                result[y * cropWidth + x] = sigmoid(sum)
            }
        }
        return Triple(result, cropWidth, cropHeight)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + Math.exp(-x.toDouble()).toFloat())

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
