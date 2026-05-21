package com.example.visionproject.calibracao.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

import com.example.visionproject.calibracao.model.CalibrationFrame;

import org.opencv.core.Point;
import org.opencv.core.Size;

import java.util.List;

/**
 * View que exibe um mapa de calor da cobertura do tabuleiro na imagem.
 * Divide a tela em uma grade 5x5 e conta quantos frames cobriram cada região.
 */
public class CoverageHeatmapView extends View {

    private static final int ROWS = 5;
    private static final int COLS = 5;
    private final int[][] grid = new int[ROWS][COLS];
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private int maxCount = 0;

    public CoverageHeatmapView(Context context) {
        super(context);
        init();
    }

    public CoverageHeatmapView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint.setStyle(Paint.Style.FILL);
    }

    public void updateCoverage(List<CalibrationFrame> frames, Size frameSize) {
        // Reinicia a grade
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                grid[i][j] = 0;
            }
        }
        maxCount = 0;

        if (frameSize == null || frameSize.width <= 0 || frameSize.height <= 0) return;

        for (CalibrationFrame f : frames) {
            Point[] pts = f.getCorners().toArray();
            if (pts.length == 0) continue;

            // Calcula o centro do tabuleiro detectado
            double avgX = 0, avgY = 0;
            for (Point p : pts) {
                avgX += p.x;
                avgY += p.y;
            }
            avgX /= pts.length;
            avgY /= pts.length;

            int col = (int) (avgX * COLS / frameSize.width);
            int row = (int) (avgY * ROWS / frameSize.height);

            if (row >= 0 && row < ROWS && col >= 0 && col < COLS) {
                grid[row][col]++;
                if (grid[row][col] > maxCount) maxCount = grid[row][col];
            }
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cellW = (float) getWidth() / COLS;
        float cellH = (float) getHeight() / ROWS;

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                int count = grid[i][j];
                
                // Gradiente de cor: Vermelho (pouco) -> Amarelo -> Verde (muito)
                int color;
                if (count == 0) {
                    color = Color.argb(50, 255, 0, 0); // Vermelho transparente
                } else {
                    float ratio = Math.min(1.0f, (float) count / 3.0f); // 3 frames por célula é considerado "bom"
                    int r = (int) (255 * (1 - ratio));
                    int g = (int) (255 * ratio);
                    color = Color.argb(150, r, g, 0);
                }

                paint.setColor(color);
                canvas.drawRect(j * cellW, i * cellH, (j + 1) * cellW, (i + 1) * cellH, paint);
            }
        }
        
        // Desenha bordas da grade
        paint.setColor(Color.WHITE);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1f);
        for (int i = 0; i <= ROWS; i++) {
            canvas.drawLine(0, i * cellH, getWidth(), i * cellH, paint);
        }
        for (int j = 0; j <= COLS; j++) {
            canvas.drawLine(j * cellW, 0, j * cellW, getHeight(), paint);
        }
        paint.setStyle(Paint.Style.FILL);
    }
}
