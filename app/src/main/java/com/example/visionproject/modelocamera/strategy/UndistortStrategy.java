package com.example.visionproject.modelocamera.strategy;

import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;

/**
 * Interface para estratégias de correção de distorção.
 * @param <INPUT> Tipo de dado de entrada.
 * @param <OUTPUT> Tipo de dado de saída.
 */
public interface UndistortStrategy<INPUT, OUTPUT> {
    OUTPUT undistort(INPUT input, CameraIntrinsics k, DistortionCoefficients d);
}
