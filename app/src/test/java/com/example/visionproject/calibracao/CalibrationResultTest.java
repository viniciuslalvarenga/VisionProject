package com.example.visionproject.calibracao;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.example.visionproject.calibracao.model.CalibrationResult;

import org.json.JSONObject;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.Date;

public class CalibrationResultTest {

    @Test
    public void testToJson() throws Exception {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            return;
        }

        Mat K = Mat.eye(3, 3, CvType.CV_64F);
        K.put(0, 0, 1400.0); // fx
        K.put(1, 1, 1400.0); // fy
        K.put(0, 2, 960.0);  // cx
        K.put(1, 2, 540.0);  // cy

        Mat D = new Mat(1, 5, CvType.CV_64F);
        D.put(0, 0, -0.3); // k1

        CalibrationResult result = new CalibrationResult(
                0.5, K, D, new ArrayList<>(), 20, 0, 1000, new Date(), new Size(1920, 1080)
        );

        JSONObject json = result.toJson();
        assertNotNull(json);
        assertEquals(0.5, json.getDouble("rms_reprojection_error_px"), 0.001);
        assertEquals(1400.0, json.getJSONObject("intrinsics").getDouble("fx"), 0.001);
        assertEquals(-0.3, json.getJSONObject("distortion").getDouble("k1"), 0.001);
        
        K.release();
        D.release();
    }
}
