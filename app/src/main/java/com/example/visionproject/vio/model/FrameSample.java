package com.example.visionproject.vio.model;

import android.graphics.Bitmap;

import org.opencv.core.Mat;

public class FrameSample {
    public final long tNs;
    public final Mat rgb;
    public final Bitmap thumb;

    public FrameSample(long tNs, Mat rgb) {
        this.tNs = tNs;
        this.rgb = rgb;
        this.thumb = null;
    }
}
