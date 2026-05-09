package com.example.visionproject.modelocamera.repository;

import com.example.visionproject.modelocamera.factory.IntrinsicsFactory;
import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;

/**
 * Singleton thread-safe que gerencia os parâmetros de calibração globais.
 */
public final class CalibrationRepository {

    private CameraIntrinsics intrinsics;
    private DistortionCoefficients distortion;

    private CalibrationRepository() {
        // Inicializa com valores padrão da especificação
        this.intrinsics = IntrinsicsFactory.defaultFor1080p();
        this.distortion = new DistortionCoefficients.Builder()
                .k1(0.12).k2(-0.08).p1(0.001).p2(0.0005).k3(0.0).build();
    }

    private static class Holder {
        static final CalibrationRepository INSTANCE = new CalibrationRepository();
    }

    public static CalibrationRepository getInstance() {
        return Holder.INSTANCE;
    }

    public synchronized CameraIntrinsics getIntrinsics() {
        return intrinsics;
    }

    public synchronized void setIntrinsics(CameraIntrinsics intrinsics) {
        this.intrinsics = intrinsics;
    }

    public synchronized DistortionCoefficients getDistortion() {
        return distortion;
    }

    public synchronized void setDistortion(DistortionCoefficients distortion) {
        this.distortion = distortion;
    }
}
