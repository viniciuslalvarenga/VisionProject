package com.example.visionproject.estereo.strategy;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.ORB;

public class OrbFeatureStrategy implements FeatureDetectionStrategy {
    private final ORB orb;

    public OrbFeatureStrategy(int nFeatures) {
        this.orb = ORB.create(nFeatures);
    }

    @Override
    public void detectAndCompute(Mat image, MatOfKeyPoint keypoints, Mat descriptors) {
        orb.detectAndCompute(image, new Mat(), keypoints, descriptors);
    }
}
