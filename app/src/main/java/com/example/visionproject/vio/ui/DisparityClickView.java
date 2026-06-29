package com.example.visionproject.vio.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.example.visionproject.vio.pipeline.DepthFromQ;

import org.opencv.core.Mat;

public class DisparityClickView extends View {
    public interface DepthClickListener {
        void onDepthQueried(double zMeters, int px, int py);
    }

    private Bitmap bitmap;
    private Mat xyzMat;
    private DepthClickListener listener;

    private final Paint crossPaint = new Paint();
    private final Paint textPaint  = new Paint();
    private float clickX = -1, clickY = -1;
    private String depthLabel = "";

    public DisparityClickView(Context ctx) { super(ctx); init(); }
    public DisparityClickView(Context ctx, AttributeSet attrs) { super(ctx, attrs); init(); }
    public DisparityClickView(Context ctx, AttributeSet attrs, int def) { super(ctx, attrs, def); init(); }

    private void init() {
        crossPaint.setColor(Color.YELLOW);
        crossPaint.setStrokeWidth(3f);
        textPaint.setColor(Color.YELLOW);
        textPaint.setTextSize(36f);
        textPaint.setAntiAlias(true);
        setOnTouchListener((v, e) -> {
            if (e.getAction() == MotionEvent.ACTION_DOWN) {
                handleClick(e.getX(), e.getY());
                return true;
            }
            return false;
        });
    }

    public void setDisparityBitmap(Bitmap bmp) {
        this.bitmap = bmp;
        invalidate();
    }

    public void setXyzMat(Mat xyz) { this.xyzMat = xyz; }

    public void setDepthClickListener(DepthClickListener l) { this.listener = l; }

    private void handleClick(float screenX, float screenY) {
        clickX = screenX; clickY = screenY;
        if (bitmap != null && xyzMat != null) {
            int imgX = (int)(screenX / getWidth()  * xyzMat.cols());
            int imgY = (int)(screenY / getHeight() * xyzMat.rows());
            double z = DepthFromQ.getZAtPixel(xyzMat, imgX, imgY);
            if (!Double.isNaN(z)) {
                depthLabel = String.format(java.util.Locale.US, "Z = %.3f m", z);
                if (listener != null) listener.onDepthQueried(z, imgX, imgY);
            } else {
                depthLabel = "Z = inválido";
            }
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap != null) {
            canvas.drawBitmap(bitmap, null,
                    new android.graphics.RectF(0, 0, getWidth(), getHeight()), null);
        }
        if (clickX >= 0) {
            float cs = 20;
            canvas.drawLine(clickX - cs, clickY, clickX + cs, clickY, crossPaint);
            canvas.drawLine(clickX, clickY - cs, clickX, clickY + cs, crossPaint);
            if (!depthLabel.isEmpty()) {
                canvas.drawText(depthLabel, clickX + 12, clickY - 12, textPaint);
            }
        }
    }
}
