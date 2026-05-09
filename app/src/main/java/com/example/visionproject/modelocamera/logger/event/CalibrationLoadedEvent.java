package com.example.visionproject.modelocamera.logger.event;

import com.example.visionproject.modelocamera.logger.SessionEvent;
import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;
import com.example.visionproject.shared.csv.CsvFormatter;

import java.util.LinkedHashMap;
import java.util.Map;

public class CalibrationLoadedEvent extends SessionEvent {
    public final CameraIntrinsics K;
    public final DistortionCoefficients D;

    public CalibrationLoadedEvent(CameraIntrinsics K, DistortionCoefficients D) {
        super("CALIBRATION_LOADED");
        this.K = K;
        this.D = D;
    }

    @Override
    public Map<String, String> toCsvFields(CsvFormatter fmt) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("fx", fmt.fmtDouble(K.getFx(), 4));
        m.put("fy", fmt.fmtDouble(K.getFy(), 4));
        m.put("cx", fmt.fmtDouble(K.getCx(), 4));
        m.put("cy", fmt.fmtDouble(K.getCy(), 4));
        m.put("k1", fmt.fmtDouble(D.getK1(), 4));
        m.put("k2", fmt.fmtDouble(D.getK2(), 4));
        m.put("p1", fmt.fmtDouble(D.getP1(), 4));
        m.put("p2", fmt.fmtDouble(D.getP2(), 4));
        m.put("k3", fmt.fmtDouble(D.getK3(), 4));
        org.opencv.core.Size res = K.getResolutionEstimate();
        m.put("res_w", String.valueOf((int)res.width));
        m.put("res_h", String.valueOf((int)res.height));
        m.put("fov_h_deg", fmt.fmtDouble(K.getHorizontalFOVDegrees((int)res.width), 2));
        m.put("distortion_type", D.getDistortionType().toString());
        m.put("notes", "Calibration parameters loaded");
        return m;
    }
}
