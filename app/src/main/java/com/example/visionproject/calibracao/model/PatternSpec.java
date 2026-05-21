package com.example.visionproject.calibracao.model;

import org.opencv.core.Mat;
import org.opencv.core.Size;

/**
 * Interface que define as especificações de um padrão de calibração.
 */
public interface PatternSpec {
    Size getInternalCornersSize();    // Size(cols, rows) — para chessboard 9x6 = (9,6)
    float getSquareSizeMm();
    Mat generateObjectPoints();       // Mat (rows*cols x 3) com Z=0 e XY em mm
    int getDetectionFlag();           // ex: Calib3d.CALIB_CB_ADAPTIVE_THRESH | NORMALIZE_IMAGE
}
