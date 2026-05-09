package com.example.visionproject.modelocamera.factory;

import com.example.visionproject.modelocamera.model.CameraIntrinsics;

/**
 * Factory para criar instâncias de CameraIntrinsics para diferentes cenários.
 */
public final class IntrinsicsFactory {

    private IntrinsicsFactory() {}

    /**
     * Valores padrão estimados para um smartphone 1080p (1920x1080).
     */
    public static CameraIntrinsics defaultFor1080p() {
        return new CameraIntrinsics(1408.5, 1410.2, 955.8, 538.1);
    }

    /**
     * Valores padrão estimados para um smartphone 4K (3840x2160).
     */
    public static CameraIntrinsics defaultFor4K() {
        // Escala os valores de 1080p por 2.0
        return new CameraIntrinsics(2817.0, 2820.4, 1911.6, 1076.2);
    }

    /**
     * Cria parâmetros intrínsecos baseados na resolução e FOV horizontal desejado.
     */
    public static CameraIntrinsics fromResolution(int w, int h, double fovHorizontalDegrees) {
        double f = (w / 2.0) / Math.tan(Math.toRadians(fovHorizontalDegrees / 2.0));
        return new CameraIntrinsics(f, f, w / 2.0, h / 2.0);
    }
}
