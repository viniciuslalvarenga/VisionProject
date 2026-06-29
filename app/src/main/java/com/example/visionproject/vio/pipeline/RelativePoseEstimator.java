package com.example.visionproject.vio.pipeline;

import com.example.visionproject.vio.model.KeyframePair;
import com.example.visionproject.vio.model.RelativePoseResult;
import com.example.visionproject.vio.strategy.OrbMatcherStrategy;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

public class RelativePoseEstimator {

    public RelativePoseResult estimate(KeyframePair pair, Mat K, Mat distCoeffs) {
        OrbMatcherStrategy.MatchResult m = pair.matches;

        List<Point> ptsA = new ArrayList<>(), ptsB = new ArrayList<>();
        List<org.opencv.core.KeyPoint> kpLList = m.kpL.toList();
        List<org.opencv.core.KeyPoint> kpRList = m.kpR.toList();
        for (DMatch d : m.good) {
            ptsA.add(kpLList.get(d.queryIdx).pt);
            ptsB.add(kpRList.get(d.trainIdx).pt);
        }

        MatOfPoint2f pA = new MatOfPoint2f();
        pA.fromList(ptsA);
        MatOfPoint2f pB = new MatOfPoint2f();
        pB.fromList(ptsB);
        MatOfPoint2f unA = new MatOfPoint2f(), unB = new MatOfPoint2f();
        Calib3d.undistortPoints(pA, unA, K, distCoeffs);
        Calib3d.undistortPoints(pB, unB, K, distCoeffs);

        Mat mask = new Mat();
        Mat E = Calib3d.findEssentialMat(unA, unB, K, Calib3d.RANSAC, 0.999, 1.0, 1000, mask);

        Mat R_rel = new Mat(), t_rel = new Mat();
        int inliers = Calib3d.recoverPose(E, unA, unB, K, R_rel, t_rel, mask);

        double angleDeg = rotationAngleDeg(R_rel);

        // Scale from IMU baseline stored in pair
        double s = pair.baseline_m;
        Mat t_metric = new Mat();
        Core.multiply(t_rel, new Scalar(s), t_metric);

        pA.release(); pB.release(); unA.release(); unB.release();
        mask.release(); E.release(); t_rel.release();

        return new RelativePoseResult(R_rel, t_metric, inliers, angleDeg, s);
    }

    private double rotationAngleDeg(Mat R) {
        double tr = R.get(0,0)[0] + R.get(1,1)[0] + R.get(2,2)[0];
        double cos = Math.max(-1, Math.min(1, (tr - 1) / 2));
        return Math.toDegrees(Math.acos(cos));
    }
}
