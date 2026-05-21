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
 * v2: Suporte a escalonamento Aspect-Fit e correção de orientação Landscape -> Portrait.
 */
public class ChessboardOverlayView extends View {

    private PatternDetectionStrategy.DetectionResult lastDetection;
    private CalibrationState currentState = CalibrationState.IDLE;
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private int frameWidth = 0;
    private int frameHeight = 0;
    
    // Optimized: reusable coordinate buffers to avoid allocation in onDraw
    private float[] cornersBuffer;
    private float[] drawCoords;

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
        linePaint.setStrokeWidth(4f);
        linePaint.setStyle(Paint.Style.STROKE);
    }

    public void setFrameSize(int width, int height) {
        this.frameWidth = width;
        this.frameHeight = height;
    }

    public void setDetection(PatternDetectionStrategy.DetectionResult detection, CalibrationState state) {
        this.lastDetection = detection;
        this.currentState = state;
        
        if (detection != null && detection.found && detection.corners != null) {
            int numPoints = (int) detection.corners.total();
            if (numPoints > 0) {
                if (cornersBuffer == null || cornersBuffer.length != numPoints * 2) {
                    cornersBuffer = new float[numPoints * 2];
                    drawCoords = new float[numPoints * 2];
                }
                // Preenche o buffer diretamente da Mat sem criar objetos Point
                detection.corners.get(0, 0, cornersBuffer);
            }
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (lastDetection == null || !lastDetection.found || cornersBuffer == null || frameWidth <= 0 || frameHeight <= 0) return;

        float viewW = getWidth();
        float viewH = getHeight();

        // Detecta se o frame está em Landscape mas a View está em Portrait
        boolean rotate = (frameWidth > frameHeight && viewH > viewW);
        
        float effFrameW = rotate ? frameHeight : frameWidth;
        float effFrameH = rotate ? frameWidth : frameHeight;

        // Calcula escala Aspect-Fit
        float scaleX = viewW / effFrameW;
        float scaleY = viewH / effFrameH;
        float scale = Math.min(scaleX, scaleY);

        // Calcula offsets para centralizar (FitCenter)
        float offsetX = (viewW - (effFrameW * scale)) / 2f;
        float offsetY = (viewH - (effFrameH * scale)) / 2f;

        // Cor baseada no estado
        int color;
        if (currentState == null) color = Color.GRAY;
        else {
            switch (currentState) {
                case POSE_STABLE: color = Color.GREEN; break;
                case DETECTING: color = Color.YELLOW; break;
                case CAPTURED: color = Color.CYAN; break;
                default: color = Color.GRAY;
            }
        }

        pointPaint.setColor(color);
        linePaint.setColor(color);

        int numPoints = cornersBuffer.length / 2;

        // Transforma coordenadas para desenho
        for (int i = 0; i < numPoints; i++) {
            float rawX = cornersBuffer[i * 2];
            float rawY = cornersBuffer[i * 2 + 1];
            float x, y;
            
            if (rotate) {
                // Rotaciona 90 graus no sentido horário: (x_l, y_l) -> (y_l, frameWidth_l - x_l)
                x = rawY;
                y = (float) (frameWidth - rawX);
            } else {
                x = rawX;
                y = rawY;
            }
            
            drawCoords[i * 2] = x * scale + offsetX;
            drawCoords[i * 2 + 1] = y * scale + offsetY;
            
            canvas.drawCircle(drawCoords[i * 2], drawCoords[i * 2 + 1], 8f, pointPaint);
        }

        // Desenha linhas conectando os pontos
        for (int i = 0; i < numPoints - 1; i++) {
            canvas.drawLine(drawCoords[i * 2], drawCoords[i * 2 + 1], 
                            drawCoords[(i + 1) * 2], drawCoords[(i + 1) * 2 + 1], linePaint);
        }
    }
}
