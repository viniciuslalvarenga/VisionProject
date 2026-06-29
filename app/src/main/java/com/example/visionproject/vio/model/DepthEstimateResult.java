package com.example.visionproject.vio.model;

import org.opencv.core.Mat;

public class DepthEstimateResult {
    public final Mat depth_xyz;
    public final double medianZ;
    public final double measuredZ_at_point;

    public DepthEstimateResult(Mat depth_xyz, double medianZ, double measuredZ_at_point) {
        this.depth_xyz = depth_xyz;
        this.medianZ = medianZ;
        this.measuredZ_at_point = measuredZ_at_point;
    }
}
