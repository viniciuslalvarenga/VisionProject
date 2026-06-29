package com.example.visionproject.vio.pipeline;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;

public class DepthFromQ {

    public static Mat reproject(Mat disp32f, Mat Q) {
        Mat xyz = new Mat();
        Calib3d.reprojectImageTo3D(disp32f, xyz, Q, true);
        return xyz;
    }

    public static double getZAtPixel(Mat xyz, int px, int py) {
        if (xyz == null || xyz.empty()) return Double.NaN;
        if (px < 0 || px >= xyz.cols() || py < 0 || py >= xyz.rows()) return Double.NaN;
        double[] v = xyz.get(py, px);
        if (v == null || v.length < 3) return Double.NaN;
        double z = v[2];
        if (Double.isInfinite(z) || Double.isNaN(z) || z <= 0 || z > 1000) return Double.NaN;
        return z;
    }

    public static double medianZ(Mat xyz) {
        int rows = xyz.rows(), cols = xyz.cols();
        java.util.List<Double> zValues = new java.util.ArrayList<>();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double[] v = xyz.get(r, c);
                if (v != null && v.length >= 3) {
                    double z = v[2];
                    if (!Double.isInfinite(z) && !Double.isNaN(z) && z > 0 && z < 1000) {
                        zValues.add(z);
                    }
                }
            }
        }
        if (zValues.isEmpty()) return Double.NaN;
        java.util.Collections.sort(zValues);
        return zValues.get(zValues.size() / 2);
    }
}
