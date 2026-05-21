package com.example.visionproject.calibracao.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import com.example.visionproject.calibracao.model.CalibrationState;
import com.example.visionproject.calibracao.strategy.PatternDetectionStrategy;

import org.opencv.core.Point;

/**
 * View para desenhar o overlay dos cantos detectados no preview da câmera.
 */
public class ChessboardOverlayView extends View {

    private PatternDetectionStrategy.DetectionResult lastDetection;
    private CalibrationState currentState = CalibrationState.IDLE;
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    public ChessboardOverlayView(Context context) {
        super(context);
        init();
    }

    public ChessboardOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        pointPaint.setStyle(Paint.Style.FILL);
        linePaint.setStrokeWidth(2f);
        linePaint.setStyle(Paint.Style.STROKE);
    }

    public void setDetection(PatternDetectionStrategy.DetectionResult detection, CalibrationState state) {
        this.lastDetection = detection;
        this.currentState = state;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (lastDetection == null || !lastDetection.found || lastDetection.corners == null) return;

        // Cor baseada no estado
        int color;
        switch (currentState) {
            case POSE_STABLE: color = Color.GREEN; break;
            case DETECTING: color = Color.YELLOW; break;
            case CAPTURED: color = Color.CYAN; break;
            default: color = Color.GRAY;
        }

        pointPaint.setColor(color);
        linePaint.setColor(color);

        Point[] points = lastDetection.corners.toArray();
        if (points.length == 0) return;

        // Desenha os cantos
        for (Point p : points) {
            canvas.drawCircle((float) p.x, (float) p.y, 6f, pointPaint);
        }

        // Desenha linhas conectando (esqueleto simplificado)
        // Assume 9x6 cantos internos (8x5 quadrados internos) se for o padrão default
        // Para simplificar, conectamos os pontos em sequência
        for (int i = 0; i < points.length - 1; i++) {
            canvas.drawLine((float) points[i].x, (float) points[i].y, 
                            (float) points[i+1].x, (float) points[i+1].y, linePaint);
        }
    }
}
