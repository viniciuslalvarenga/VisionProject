package com.example.visionproject.modelocamera.strategy;

import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;

/**
 * Estratégia para corrigir a distorção de uma imagem inteira.
 */
public final class ImageUndistortStrategy implements UndistortStrategy<Mat, Mat> {

    @Override
    public Mat undistort(Mat input, CameraIntrinsics k, DistortionCoefficients d) {
        if (input == null || input.empty()) {
            return new Mat();
        }
        Mat dst = new Mat();
        Calib3d.undistort(input, dst, k.toMat(), d.toMat());
        return dst;
    }
}
