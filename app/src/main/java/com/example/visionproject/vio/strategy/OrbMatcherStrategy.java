package com.example.visionproject.vio.strategy;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class OrbMatcherStrategy {

    public static class MatchResult {
        public MatOfKeyPoint kpL, kpR;
        public Mat undL, undR;
        public List<DMatch> good;
    }

    public MatchResult match(Mat imgL, Mat imgR, Mat K, Mat distCoeffs) {
        Mat undL = new Mat(), undR = new Mat();
        Calib3d.undistort(imgL, undL, K, distCoeffs);
        Calib3d.undistort(imgR, undR, K, distCoeffs);

        Mat grayL = new Mat(), grayR = new Mat();
        Imgproc.cvtColor(undL, grayL, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(undR, grayR, Imgproc.COLOR_BGR2GRAY);

        ORB orb = ORB.create(2000);
        MatOfKeyPoint kpL = new MatOfKeyPoint(), kpR = new MatOfKeyPoint();
        Mat desL = new Mat(), desR = new Mat();
        orb.detectAndCompute(grayL, new Mat(), kpL, desL);
        orb.detectAndCompute(grayR, new Mat(), kpR, desR);

        List<DMatch> good = new ArrayList<>();
        if (!desL.empty() && !desR.empty()) {
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            List<MatOfDMatch> knn = new ArrayList<>();
            matcher.knnMatch(desL, desR, knn, 2);
            for (MatOfDMatch m : knn) {
                DMatch[] arr = m.toArray();
                if (arr.length >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                    good.add(arr[0]);
                }
            }
        }

        grayL.release(); grayR.release(); desL.release(); desR.release();

        MatchResult r = new MatchResult();
        r.kpL = kpL; r.kpR = kpR; r.undL = undL; r.undR = undR; r.good = good;
        return r;
    }
}
