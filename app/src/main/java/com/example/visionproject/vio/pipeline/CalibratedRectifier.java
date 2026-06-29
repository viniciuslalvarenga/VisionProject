package com.example.visionproject.vio.pipeline;

import com.example.visionproject.vio.model.CalibratedRectifyResult;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class CalibratedRectifier {

    public CalibratedRectifyResult rectify(Mat imgL, Mat imgR, Mat K, Mat distCoeffs,
                                            Mat R_rel, Mat t_metric, Size imgSize) {
        Mat R1 = new Mat(), R2 = new Mat();
        Mat P1 = new Mat(), P2 = new Mat();
        Mat Q  = new Mat();
        Calib3d.stereoRectify(K, distCoeffs, K, distCoeffs, imgSize,
                R_rel, t_metric, R1, R2, P1, P2, Q,
                Calib3d.CALIB_ZERO_DISPARITY, 0);

        Mat m1x = new Mat(), m1y = new Mat();
        Mat m2x = new Mat(), m2y = new Mat();
        Calib3d.initUndistortRectifyMap(K, distCoeffs, R1, P1, imgSize,
                CvType.CV_32FC1, m1x, m1y);
        Calib3d.initUndistortRectifyMap(K, distCoeffs, R2, P2, imgSize,
                CvType.CV_32FC1, m2x, m2y);

        Mat rectL = new Mat(), rectR = new Mat();
        Imgproc.remap(imgL, rectL, m1x, m1y, Imgproc.INTER_LINEAR);
        Imgproc.remap(imgR, rectR, m2x, m2y, Imgproc.INTER_LINEAR);

        m1x.release(); m1y.release(); m2x.release(); m2y.release();
        return new CalibratedRectifyResult(rectL, rectR, R1, R2, P1, P2, Q);
    }
}
