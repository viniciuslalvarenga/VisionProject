package com.example.visionproject.estereo.model;

import java.io.Serializable;

public class SgbmParams implements Serializable {
    public final int minDisparity;
    public final int numDisparities;
    public final int blockSize;
    public final int P1;
    public final int P2;
    public final int disp12MaxDiff;
    public final int preFilterCap;
    public final int uniquenessRatio;
    public final int speckleWindowSize;
    public final int speckleRange;

    private SgbmParams(Builder builder) {
        this.minDisparity = builder.minDisparity;
        this.numDisparities = builder.numDisparities;
        this.blockSize = builder.blockSize;
        this.P1 = builder.P1;
        this.P2 = builder.P2;
        this.disp12MaxDiff = builder.disp12MaxDiff;
        this.preFilterCap = builder.preFilterCap;
        this.uniquenessRatio = builder.uniquenessRatio;
        this.speckleWindowSize = builder.speckleWindowSize;
        this.speckleRange = builder.speckleRange;
    }

    public static class Builder {
        private int minDisparity = 0;
        private int numDisparities = 64;
        private int blockSize = 5;
        private int P1 = 0;
        private int P2 = 0;
        private int disp12MaxDiff = 1;
        private int preFilterCap = 63;
        private int uniquenessRatio = 10;
        private int speckleWindowSize = 100;
        private int speckleRange = 2;

        public Builder minDisparity(int val) { minDisparity = val; return this; }
        public Builder numDisparities(int val) { numDisparities = val; return this; }
        public Builder blockSize(int val) { blockSize = val; return this; }
        public Builder disp12MaxDiff(int val) { disp12MaxDiff = val; return this; }
        public Builder preFilterCap(int val) { preFilterCap = val; return this; }
        public Builder uniquenessRatio(int val) { uniquenessRatio = val; return this; }
        public Builder speckleWindowSize(int val) { speckleWindowSize = val; return this; }
        public Builder speckleRange(int val) { speckleRange = val; return this; }

        public SgbmParams build() {
            if (P1 == 0) P1 = 8 * 3 * blockSize * blockSize;
            if (P2 == 0) P2 = 32 * 3 * blockSize * blockSize;
            return new SgbmParams(this);
        }
    }
}
