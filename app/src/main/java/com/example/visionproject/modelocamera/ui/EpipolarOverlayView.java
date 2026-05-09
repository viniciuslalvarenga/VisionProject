package com.example.visionproject.modelocamera.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.Nullable;

import com.example.visionproject.modelocamera.model.EpipolarPoint;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

/**
 * Custom View para marcação de pontos e exibição de linhas epipolares.
 */
public class EpipolarOverlayView extends View {

    private Bitmap backgroundImage;
    private final List<EpipolarPoint> points = new ArrayList<>();
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Rect destRect = new Rect();

    private static final int[] COLORS = {
            Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.MAGENTA
    };

    public EpipolarOverlayView(Context context) {
        super(context);
        init();
    }

    public EpipolarOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        pointPaint.setStyle(Paint.Style.FILL);
        linePaint.setStrokeWidth(3f);
    }

    public void setBackgroundImage(Bitmap bitmap) {
        this.backgroundImage = bitmap;
        invalidate();
    }

    public void clearPoints() {
        points.clear();
        invalidate();
    }

    public List<EpipolarPoint> getPoints() {
        return new ArrayList<>(points);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (backgroundImage != null) {
            calculateDestRect();
            canvas.drawBitmap(backgroundImage, null, destRect, null);
        }

        for (EpipolarPoint ep : points) {
            Point p = ep.getCoords();
            float screenX = (float) (p.x * destRect.width() / (backgroundImage != null ? backgroundImage.getWidth() : 1) + destRect.left);
            float screenY = (float) (p.y * destRect.height() / (backgroundImage != null ? backgroundImage.getHeight() : 1) + destRect.top);

            pointPaint.setColor(ep.getColorRGB());
            canvas.drawCircle(screenX, screenY, 15f, pointPaint);

            linePaint.setColor(ep.getColorRGB());
            // Desenha linha horizontal (epipolar simplificada)
            canvas.drawLine(destRect.left, screenY, destRect.right, screenY, linePaint);
        }
    }

    private void calculateDestRect() {
        if (backgroundImage == null) return;
        float viewWidth = getWidth();
        float viewHeight = getHeight();
        float imgWidth = backgroundImage.getWidth();
        float imgHeight = backgroundImage.getHeight();

        float scale = Math.min(viewWidth / imgWidth, viewHeight / imgHeight);
        float dw = imgWidth * scale;
        float dh = imgHeight * scale;

        destRect.set(
                (int) ((viewWidth - dw) / 2),
                (int) ((viewHeight - dh) / 2),
                (int) ((viewWidth + dw) / 2),
                (int) ((viewHeight + dh) / 2)
        );
    }

    private OnPointAddedListener listener;

    public interface OnPointAddedListener {
        void onPointAdded(Point p, int index, int color);
    }

    public void setOnPointAddedListener(OnPointAddedListener listener) {
        this.listener = listener;
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN && backgroundImage != null) {
            performClick();
            float x = event.getX();
            float y = event.getY();

            if (destRect.contains((int) x, (int) y)) {
                // Converte para coordenadas da imagem
                double imgX = (x - destRect.left) * backgroundImage.getWidth() / destRect.width();
                double imgY = (y - destRect.top) * backgroundImage.getHeight() / destRect.height();

                if (points.size() >= 5) {
                    points.remove(0); // Remove o mais antigo
                }

                int color = COLORS[points.size() % COLORS.length];
                int index = points.size();
                Point p = new Point(imgX, imgY);
                points.add(new EpipolarPoint(p, color, index));
                
                if (listener != null) {
                    listener.onPointAdded(p, index, color);
                }

                invalidate();
                return true;
            }
        }
        return super.onTouchEvent(event);
    }
}
