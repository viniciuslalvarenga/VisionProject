package com.example.visionproject.calibracao.strategy;

import com.example.visionproject.calibracao.model.PatternSpec;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;

/**
 * Interface para estratégias de detecção de padrões de calibração.
 */
public interface PatternDetectionStrategy {
    class DetectionResult {
        public final boolean found;
        public final MatOfPoint2f corners;
        public final long elapsedMs;

        public DetectionResult(boolean found, MatOfPoint2f corners, long elapsedMs) {
            this.found = found;
            this.corners = corners;
            this.elapsedMs = elapsedMs;
        }
    }

    DetectionResult detect(Mat grayFrame, PatternSpec pattern);
}
