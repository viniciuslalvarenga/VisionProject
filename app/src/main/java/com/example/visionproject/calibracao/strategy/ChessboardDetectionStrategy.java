package com.example.visionproject.calibracao.strategy;

import com.example.visionproject.calibracao.model.PatternSpec;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

/**
 * Estratégia de detecção para tabuleiros de xadrez (Chessboard).
 */
public class ChessboardDetectionStrategy implements PatternDetectionStrategy {

    @Override
    public DetectionResult detect(Mat grayFrame, PatternSpec pattern) {
        long t0 = System.currentTimeMillis();
        MatOfPoint2f corners = new MatOfPoint2f();
        boolean found = Calib3d.findChessboardCorners(
                grayFrame,
                pattern.getInternalCornersSize(),
                corners,
                pattern.getDetectionFlag()
        );

        if (found) {
            // Refinamento de subpixel para maior precisão (Zhang exige alta precisão)
            TermCriteria tc = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 30, 0.001);
            Imgproc.cornerSubPix(grayFrame, corners, new Size(11, 11), new Size(-1, -1), tc);
        } else {
            corners.release();
            corners = null;
        }

        long elapsed = System.currentTimeMillis() - t0;
        return new DetectionResult(found, corners, elapsed);
    }
}
