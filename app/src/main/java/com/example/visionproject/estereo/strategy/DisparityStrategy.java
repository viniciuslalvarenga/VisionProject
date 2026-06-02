package com.example.visionproject.estereo.strategy;

import com.example.visionproject.estereo.model.DisparityResult;
import org.opencv.core.Mat;

public interface DisparityStrategy {
    DisparityResult compute(Mat grayL, Mat grayR);
}
