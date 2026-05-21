package com.example.visionproject.calibracao.model;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

/**
 * Implementação de PatternSpec para um tabuleiro de xadrez (Chessboard).
 */
public class ChessboardSpec implements PatternSpec {
    private final int cols;
    private final int rows;
    private final float squareSizeMm;

    public ChessboardSpec(int cols, int rows, float squareSizeMm) {
        if (cols < 3 || rows < 3) {
            throw new IllegalArgumentException("Tabuleiro deve ter pelo menos 3x3 cantos internos");
        }
        if (squareSizeMm <= 0) {
            throw new IllegalArgumentException("Tamanho do quadrado deve ser positivo");
        }
        this.cols = cols;
        this.rows = rows;
        this.squareSizeMm = squareSizeMm;
    }

    @Override
    public Size getInternalCornersSize() {
        return new Size(cols, rows);
    }

    @Override
    public float getSquareSizeMm() {
        return squareSizeMm;
    }

    @Override
    public Mat generateObjectPoints() {
        // Gera grade 3D Z=0: (0,0,0), (s,0,0), (2s,0,0)...
        Mat obj = new Mat(rows * cols, 1, CvType.CV_32FC3);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                obj.put(i * cols + j, 0, new float[]{j * squareSizeMm, i * squareSizeMm, 0.0f});
            }
        }
        return obj;
    }

    @Override
    public int getDetectionFlag() {
        return Calib3d.CALIB_CB_ADAPTIVE_THRESH | Calib3d.CALIB_CB_NORMALIZE_IMAGE | Calib3d.CALIB_CB_FAST_CHECK;
    }
}
