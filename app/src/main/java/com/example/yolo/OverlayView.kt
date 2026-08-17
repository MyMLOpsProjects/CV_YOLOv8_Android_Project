package com.example.yolo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = mutableListOf<BoundingBox>()
    private var lastInferenceTime: Long = 0
    
    private val boxPaint = Paint()
    private val textBackgroundPaint = Paint()
    private val textPaint = Paint()
    private val timePaint = Paint()

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
        textBackgroundPaint.color = Color.parseColor("#99000000") // Semi-transparent black
        textBackgroundPaint.style = Paint.Style.FILL

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 40f
        textPaint.isFakeBoldText = true

        boxPaint.color = Color.GREEN // Changed to Green for better visibility
        boxPaint.strokeWidth = 6F
        boxPaint.style = Paint.Style.STROKE
        
        timePaint.color = Color.YELLOW
        timePaint.textSize = 45f
        timePaint.isFakeBoldText = true
        timePaint.setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results.forEach {
            val left = it.x1 * width
            val top = it.y1 * height
            val right = it.x2 * width
            val bottom = it.y2 * height

            // Draw bounding box
            canvas.drawRect(left, top, right, bottom, boxPaint)
            
            // Prepare label text
            val drawableText = "${it.clsName} ${"%.2f".format(it.cnf)}"
            textPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            
            // Draw background for text
            canvas.drawRect(
                left,
                top - textHeight - 16,
                left + textWidth + 16,
                top,
                textBackgroundPaint
            )
            
            // Draw label text
            canvas.drawText(drawableText, left + 8, top - 8, textPaint)
        }
        
        // Draw inference time at the bottom
        if (lastInferenceTime > 0) {
            val timeText = "Inference: ${lastInferenceTime}ms"
            canvas.drawText(timeText, 40f, height - 40f, timePaint)
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        results = boundingBoxes.toMutableList()
        lastInferenceTime = inferenceTime
        invalidate()
    }
}
