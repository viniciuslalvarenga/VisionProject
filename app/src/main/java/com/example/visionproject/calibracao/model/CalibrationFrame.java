package com.example.visionproject.calibracao.model;

import org.opencv.core.MatOfPoint2f;

/**
 * Modelo imutável que representa um frame capturado para calibração.
 */
public class CalibrationFrame {
    private final int index;
    private final long timestampMs;
    private final MatOfPoint2f corners;
    private final int cornersFound;
    private final double blurScore;
    private final double poseStabilityScore;

    public CalibrationFrame(int index, long timestampMs, MatOfPoint2f corners, int cornersFound, double blurScore, double poseStabilityScore) {
        this.index = index;
        this.timestampMs = timestampMs;
        this.corners = corners;
        this.cornersFound = cornersFound;
        this.blurScore = blurScore;
        this.poseStabilityScore = poseStabilityScore;
    }

    public int getIndex() { return index; }
    public long getTimestampMs() { return timestampMs; }
    public MatOfPoint2f getCorners() { return corners; }
    public int getCornersFound() { return cornersFound; }
    public double getBlurScore() { return blurScore; }
    public double getPoseStabilityScore() { return poseStabilityScore; }

    public void release() {
        if (corners != null) {
            corners.release();
        }
    }
}
