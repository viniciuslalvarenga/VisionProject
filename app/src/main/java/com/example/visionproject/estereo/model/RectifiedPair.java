package com.example.visionproject.estereo.model;

import org.opencv.core.Mat;

public class RectifiedPair {
    public final Mat rectL;
    public final Mat rectR;
    public final Mat H1;
    public final Mat H2;

    public RectifiedPair(Mat rectL, Mat rectR, Mat H1, Mat H2) {
        this.rectL = rectL;
        this.rectR = rectR;
        this.H1 = H1;
        this.H2 = H2;
    }

    public void release() {
        if (rectL != null) rectL.release();
        if (rectR != null) rectR.release();
        if (H1 != null) H1.release();
        if (H2 != null) H2.release();
    }
}
