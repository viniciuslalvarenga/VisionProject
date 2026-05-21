package com.example.visionproject.calibracao.strategy;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;

/**
 * Estratégia para detecção de desfoque (Blur) usando variância do Laplaciano.
 */
public class BlurDetectionStrategy {
    private double threshold = 100.0;

    public boolean isSharp(Mat gray) {
        return getScore(gray) >= threshold;
    }

    public double getScore(Mat gray) {
        Mat lap = new Mat();
        Imgproc.Laplacian(gray, lap, CvType.CV_64F);
        MatOfDouble mean = new MatOfDouble(), stdDev = new MatOfDouble();
        Core.meanStdDev(lap, mean, stdDev);
        double variance = Math.pow(stdDev.get(0, 0)[0], 2);
        lap.release();
        return variance;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
