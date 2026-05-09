package com.example.visionproject.modelocamera.model;

import org.opencv.core.MatOfDouble;
import java.util.Objects;

/**
 * Classe imutável que representa os coeficientes de distorção da câmera (k1, k2, p1, p2, k3).
 */
public final class DistortionCoefficients {
    private final double k1;
    private final double k2;
    private final double p1;
    private final double p2;
    private final double k3;

    public enum DistortionType {
        BARREL, PINCUSHION, NONE
    }

    private DistortionCoefficients(Builder builder) {
        this.k1 = builder.k1;
        this.k2 = builder.k2;
        this.p1 = builder.p1;
        this.p2 = builder.p2;
        this.k3 = builder.k3;
    }

    public double getK1() { return k1; }
    public double getK2() { return k2; }
    public double getP1() { return p1; }
    public double getP2() { return p2; }
    public double getK3() { return k3; }

    /**
     * Retorna os coeficientes no formato OpenCV (MatOfDouble com 5 elementos).
     */
    public MatOfDouble toMat() {
        return new MatOfDouble(k1, k2, p1, p2, k3);
    }

    /**
     * Retorna o tipo de distorção baseado no sinal de k1.
     */
    public DistortionType getDistortionType() {
        if (k1 < -0.05) return DistortionType.BARREL;
        if (k1 > 0.05) return DistortionType.PINCUSHION;
        return DistortionType.NONE;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DistortionCoefficients that = (DistortionCoefficients) o;
        return Double.compare(that.k1, k1) == 0 &&
                Double.compare(that.k2, k2) == 0 &&
                Double.compare(that.p1, p1) == 0 &&
                Double.compare(that.p2, p2) == 0 &&
                Double.compare(that.k3, k3) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(k1, k2, p1, p2, k3);
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.US, "D[k1=%.4f, k2=%.4f, p1=%.4f, p2=%.4f, k3=%.4f]", k1, k2, p1, p2, k3);
    }

    public static class Builder {
        private double k1 = 0.0;
        private double k2 = 0.0;
        private double p1 = 0.0;
        private double p2 = 0.0;
        private double k3 = 0.0;

        public Builder k1(double val) { k1 = val; return this; }
        public Builder k2(double val) { k2 = val; return this; }
        public Builder p1(double val) { p1 = val; return this; }
        public Builder p2(double val) { p2 = val; return this; }
        public Builder k3(double val) { k3 = val; return this; }

        public DistortionCoefficients build() {
            return new DistortionCoefficients(this);
        }
    }
}
