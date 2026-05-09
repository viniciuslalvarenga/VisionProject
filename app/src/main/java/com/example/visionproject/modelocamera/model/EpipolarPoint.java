package com.example.visionproject.modelocamera.model;

import org.opencv.core.Point;
import java.util.Objects;

/**
 * POJO imutável que representa um ponto marcado para visualização epipolar.
 */
public final class EpipolarPoint {
    private final Point coords;
    private final int colorRGB;
    private final int index;

    public EpipolarPoint(Point coords, int colorRGB, int index) {
        this.coords = new Point(coords.x, coords.y);
        this.colorRGB = colorRGB;
        this.index = index;
    }

    public Point getCoords() {
        return new Point(coords.x, coords.y);
    }

    public int getColorRGB() {
        return colorRGB;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EpipolarPoint that = (EpipolarPoint) o;
        return colorRGB == that.colorRGB &&
                index == that.index &&
                Objects.equals(coords.x, that.coords.x) &&
                Objects.equals(coords.y, that.coords.y);
    }

    @Override
    public int hashCode() {
        return Objects.hash(coords.x, coords.y, colorRGB, index);
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.US, "EpipolarPoint[index=%d, x=%.1f, y=%.1f]", index, coords.x, coords.y);
    }
}
