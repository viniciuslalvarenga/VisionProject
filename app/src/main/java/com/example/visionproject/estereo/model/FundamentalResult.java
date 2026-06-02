package com.example.visionproject.estereo.model;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import java.util.List;

public class FundamentalResult {
    public final Mat F;
    public final int inliers;
    public final Mat mask;
    public final List<Point> inliersL;
    public final List<Point> inliersR;

    public FundamentalResult(Mat F, int inliers, Mat mask, List<Point> inliersL, List<Point> inliersR) {
        this.F = F;
        this.inliers = inliers;
        this.mask = mask;
        this.inliersL = inliersL;
        this.inliersR = inliersR;
    }
}
