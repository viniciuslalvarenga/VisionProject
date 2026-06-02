package com.example.visionproject.estereo.strategy;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public interface FeatureDetectionStrategy {
    void detectAndCompute(Mat image, MatOfKeyPoint keypoints, Mat descriptors);
}
