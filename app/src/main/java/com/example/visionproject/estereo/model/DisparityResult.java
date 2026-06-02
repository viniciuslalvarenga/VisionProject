package com.example.visionproject.estereo.model;

import org.opencv.core.Mat;

public class DisparityResult {
    public final Mat disp32f;
    public final SgbmParams params;
    public final long elapsedMs;

    public DisparityResult(Mat disp32f, SgbmParams params, long elapsedMs) {
        this.disp32f = disp32f;
        this.params = params;
        this.elapsedMs = elapsedMs;
    }
}
