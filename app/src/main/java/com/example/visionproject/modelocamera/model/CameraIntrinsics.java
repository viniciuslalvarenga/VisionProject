package com.example.visionproject.modelocamera.model;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.util.Objects;

/**
 * Classe imutável que representa a matriz de parâmetros intrínsecos (K) da câmera.
 */
public final class CameraIntrinsics {
    private final double fx;
    private final double fy;
    private final double cx;
    private final double cy;
    private final double s; // Skew

    public CameraIntrinsics(double fx, double fy, double cx, double cy) {
        this(fx, fy, cx, cy, 0.0);
    }

    public CameraIntrinsics(double fx, double fy, double cx, double cy, double s) {
        if (fx <= 0 || fy <= 0) {
            throw new IllegalArgumentException("fx e fy devem ser maiores que zero.");
        }
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
        this.s = s;
    }

    public double getFx() { return fx; }
    public double getFy() { return fy; }
    public double getCx() { return cx; }
    public double getCy() { return cy; }
    public double getS() { return s; }

    /**
     * Retorna a matriz K no formato OpenCV (3x3, CV_64F).
     */
    public Mat toMat() {
        Mat k = new Mat(3, 3, CvType.CV_64F);
        k.put(0, 0, fx, s, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        return k;
    }

    /**
     * Calcula o FOV horizontal em graus.
     */
    public double getHorizontalFOVDegrees(int imageWidth) {
        return Math.toDegrees(2 * Math.atan(imageWidth / (2 * fx)));
    }

    /**
     * Retorna uma estimativa da resolução baseada no ponto principal.
     */
    public Size getResolutionEstimate() {
        return new Size(2 * cx, 2 * cy);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CameraIntrinsics that = (CameraIntrinsics) o;
        return Double.compare(that.fx, fx) == 0 &&
                Double.compare(that.fy, fy) == 0 &&
                Double.compare(that.cx, cx) == 0 &&
                Double.compare(that.cy, cy) == 0 &&
                Double.compare(that.s, s) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(fx, fy, cx, cy, s);
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.US, "K[fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, s=%.1f]", fx, fy, cx, cy, s);
    }
}
