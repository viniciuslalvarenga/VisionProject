package com.example.visionproject.estereo.factory;

import com.example.visionproject.estereo.model.SgbmParams;

public class SgbmParamsFactory {
    public static SgbmParams fast() {
        return new SgbmParams.Builder()
                .minDisparity(-128)
                .numDisparities(256)
                .blockSize(5)
                .uniquenessRatio(10)
                .speckleWindowSize(100)
                .speckleRange(2)
                .build();
    }

    public static SgbmParams balanced() {
        return new SgbmParams.Builder()
                .minDisparity(-256)
                .numDisparities(512)
                .blockSize(5)
                .uniquenessRatio(10)
                .speckleWindowSize(100)
                .speckleRange(2)
                .build();
    }

    public static SgbmParams quality() {
        return new SgbmParams.Builder()
                .minDisparity(-384)
                .numDisparities(768)
                .blockSize(5)
                .uniquenessRatio(15)
                .speckleWindowSize(150)
                .speckleRange(2)
                .build();
    }
}
