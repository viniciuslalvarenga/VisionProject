package com.example.visionproject.vio.model;

import org.opencv.core.Mat;

public class CalibratedRectifyResult {
    public final Mat rectL;
    public final Mat rectR;
    public final Mat R1;
    public final Mat R2;
    public final Mat P1;
    public final Mat P2;
    public final Mat Q;

    public CalibratedRectifyResult(Mat rectL, Mat rectR, Mat R1, Mat R2, Mat P1, Mat P2, Mat Q) {
        this.rectL = rectL;
        this.rectR = rectR;
        this.R1 = R1;
        this.R2 = R2;
        this.P1 = P1;
        this.P2 = P2;
        this.Q = Q;
    }
}
