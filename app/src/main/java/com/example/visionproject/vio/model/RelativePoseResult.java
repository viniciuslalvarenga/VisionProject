package com.example.visionproject.vio.model;

import org.opencv.core.Mat;

public class RelativePoseResult {
    public final Mat R_rel;
    public final Mat t_metric;
    public final int inliers;
    public final double rotAngleDeg;
    public final double scale_m;

    public RelativePoseResult(Mat R_rel, Mat t_metric, int inliers, double rotAngleDeg, double scale_m) {
        this.R_rel = R_rel;
        this.t_metric = t_metric;
        this.inliers = inliers;
        this.rotAngleDeg = rotAngleDeg;
        this.scale_m = scale_m;
    }
}
