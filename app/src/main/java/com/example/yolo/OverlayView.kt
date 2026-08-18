package com.example.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = mutableListOf<BoundingBox>()
    private var lastInferenceTime: Long = 0
    
    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()
    private val timePaint = Paint()
    private val maskPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        results.clear()
        lastInferenceTime = 0
        invalidate()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.parseColor("#99000000")
        textBackgroundPaint.style = Paint.Style.FILL

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 40f
        textPaint.isFakeBoldText = true

        boxPaint.color = Color.GREEN
        boxPaint.strokeWidth = 6F
        boxPaint.style = Paint.Style.STROKE
        
        timePaint.color = Color.YELLOW
        timePaint.textSize = 45f
        timePaint.isFakeBoldText = true
        timePaint.setShadowLayer(5f, 0f, 0f, Color.BLACK)

        maskPaint.alpha = 140 // Slightly higher opacity for segmentation
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // 1. Draw Mask if available
            it.mask?.let { mask ->
                if (it.mWidth > 0 && it.mHeight > 0) {
                    drawMask(canvas, mask, it.mWidth, it.mHeight, left, top, right, bottom, it.cls)
                }
            }

            // 2. Draw bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint)
            
            // 3. Prepare label text
            val drawableText = "${it.clsName} ${"%.2f".format(it.cnf)}"
            textPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            
            canvas.drawRect(
                left,
                top - textHeight - 16,
                left + textWidth + 16,
                top,
                textBackgroundPaint
            )
            
            canvas.drawText(drawableText, left + 8, top - 8, textPaint)
        }
        
        if (lastInferenceTime > 0) {
            val timeText = "Inference: ${lastInferenceTime}ms"
            canvas.drawText(timeText, 40f, height - 40f, timePaint)
        }
    }

    private fun drawMask(canvas: Canvas, mask: FloatArray, mWidth: Int, mHeight: Int, left: Float, top: Float, right: Float, bottom: Float, classId: Int) {
        val maskBitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.ARGB_8888)
        val maskColors = IntArray(mask.size)

        val color = getColorForClass(classId)
        val baseR = Color.red(color)
        val baseG = Color.green(color)
        val baseB = Color.blue(color)

        for (i in mask.indices) {
            // YOLOv8 segmentation masks are binary after sigmoid
            val alpha = if (mask[i] > 0.5f) 120 else 0 
            maskColors[i] = Color.argb(alpha, baseR, baseG, baseB)
        }
        
        maskBitmap.setPixels(maskColors, 0, mWidth, 0, 0, mWidth, mHeight)
        
        val destRect = RectF(left, top, right, bottom)
        canvas.drawBitmap(maskBitmap, null, destRect, maskPaint)
        
        // Recycle bitmap to save memory, though for small 160x160 crops GC usually handles it
        // In a production app, we would use a pool.
    }

    private fun getColorForClass(id: Int): Int {
        val colors = intArrayOf(
            Color.RED, Color.BLUE, Color.MAGENTA, Color.YELLOW, Color.CYAN, 
            Color.GREEN, Color.parseColor("#FFA500"), Color.parseColor("#800080")
        )
        return colors[id % colors.size]
    }

    fun setResults(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        results = boundingBoxes.toMutableList()
        lastInferenceTime = inferenceTime
        invalidate()
    }
}
