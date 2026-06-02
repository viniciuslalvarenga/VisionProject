package com.example.visionproject.estereo.strategy;

import com.example.visionproject.estereo.model.DisparityResult;
import com.example.visionproject.estereo.model.SgbmParams;
import org.opencv.calib3d.StereoSGBM;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class SgbmDisparityStrategy implements DisparityStrategy {
    private final SgbmParams params;

    public SgbmDisparityStrategy(SgbmParams params) {
        this.params = params;
    }

    @Override
    public DisparityResult compute(Mat grayL, Mat grayR) {
        long t0 = System.currentTimeMillis();

        StereoSGBM sgbm = StereoSGBM.create(
                params.minDisparity,
                params.numDisparities,
                params.blockSize,
                params.P1,
                params.P2,
                params.disp12MaxDiff,
                params.preFilterCap,
                params.uniquenessRatio,
                params.speckleWindowSize,
                params.speckleRange,
                StereoSGBM.MODE_SGBM
        );

        Mat disp16 = new Mat();
        sgbm.compute(grayL, grayR, disp16);

        Mat disp32f = new Mat();
        disp16.convertTo(disp32f, CvType.CV_32F, 1.0 / 16.0);

        long elapsed = System.currentTimeMillis() - t0;
        disp16.release();

        return new DisparityResult(disp32f, params, elapsed);
    }
}
