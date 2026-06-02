package com.example.visionproject.estereo.pipeline;

import com.example.visionproject.estereo.model.FundamentalResult;
import com.example.visionproject.estereo.model.RectifiedPair;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class RectificationEstimator {
    public static RectifiedPair rectify(Mat undL, Mat undR, FundamentalResult fr, Mat K) {
        try {
            return rectifyWithIntrinsics(undL, undR, fr, K);
        } catch (Exception ex) {
            return rectifyUncalibrated(undL, undR, fr);
        }
    }

    private static RectifiedPair rectifyWithIntrinsics(Mat undL, Mat undR, FundamentalResult fr, Mat K) {
        Size size = undL.size();
        MatOfPoint2f inL = new MatOfPoint2f();
        MatOfPoint2f inR = new MatOfPoint2f();
        inL.fromList(fr.inliersL);
        inR.fromList(fr.inliersR);

        Mat E = new Mat();
        Mat tmp = new Mat();
        Core.gemm(K.t(), fr.F, 1.0, new Mat(), 0.0, tmp);
        Core.gemm(tmp, K, 1.0, new Mat(), 0.0, E);
        tmp.release();

        Mat R = new Mat();
        Mat t = new Mat();
        Mat mask = new Mat();
        Calib3d.recoverPose(E, inL, inR, K, R, t, mask);
        E.release();
        mask.release();

        Mat R1 = new Mat();
        Mat R2 = new Mat();
        Mat P1 = new Mat();
        Mat P2 = new Mat();
        Mat Q = new Mat();
        Rect roi1 = new Rect();
        Rect roi2 = new Rect();
        Calib3d.stereoRectify(K, new Mat(), K, new Mat(), size, R, t,
                R1, R2, P1, P2, Q,
                Calib3d.CALIB_ZERO_DISPARITY, 0, size,
                roi1, roi2);
        Q.release();

        Mat map1x = new Mat();
        Mat map1y = new Mat();
        Mat map2x = new Mat();
        Mat map2y = new Mat();
        Calib3d.initUndistortRectifyMap(K, new Mat(), R1, P1, size, CvType.CV_32FC1, map1x, map1y);
        Calib3d.initUndistortRectifyMap(K, new Mat(), R2, P2, size, CvType.CV_32FC1, map2x, map2y);

        Mat rectL = new Mat();
        Mat rectR = new Mat();
        Imgproc.remap(undL, rectL, map1x, map1y, Imgproc.INTER_LINEAR);
        Imgproc.remap(undR, rectR, map2x, map2y, Imgproc.INTER_LINEAR);

        map1x.release(); map1y.release(); map2x.release(); map2y.release();
        P1.release(); P2.release();
        return new RectifiedPair(rectL, rectR, R1, R2);
    }

    private static RectifiedPair rectifyUncalibrated(Mat undL, Mat undR, FundamentalResult fr) {
        Size size = undL.size();
        MatOfPoint2f inL = new MatOfPoint2f();
        MatOfPoint2f inR = new MatOfPoint2f();
        inL.fromList(fr.inliersL);
        inR.fromList(fr.inliersR);

        Mat H1 = new Mat();
        Mat H2 = new Mat();
        boolean ok = Calib3d.stereoRectifyUncalibrated(inL, inR, fr.F, size, H1, H2, 5.0);
        if (!ok) {
            H1.release();
            H2.release();
            throw new RuntimeException("stereoRectifyUncalibrated falhou em encontrar uma solução.");
        }

        Mat rectL = new Mat();
        Mat rectR = new Mat();
        Imgproc.warpPerspective(undL, rectL, H1, size);
        Imgproc.warpPerspective(undR, rectR, H2, size);

        return new RectifiedPair(rectL, rectR, H1, H2);
    }
}
