package com.example.visionproject.calibracao.pipeline;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Analisador de erro de reprojeção para validar a qualidade da calibração.
 */
public class ReprojectionErrorAnalyzer {

    public static List<Double> computePerImage(List<Mat> objectPoints, List<Mat> imagePoints,
                                              List<Mat> rvecs, List<Mat> tvecs,
                                              Mat cameraMatrix, Mat distCoeffs) {
        List<Double> errors = new ArrayList<>();
        MatOfPoint2f projectedPoints = new MatOfPoint2f();
        MatOfDouble distCoeffsDouble = new MatOfDouble(distCoeffs);

        for (int i = 0; i < objectPoints.size(); i++) {
            Mat obj = objectPoints.get(i);
            MatOfPoint3f obj3f = new MatOfPoint3f(obj);
            Calib3d.projectPoints(obj3f, rvecs.get(i), tvecs.get(i),
                    cameraMatrix, distCoeffsDouble, projectedPoints);

            double error = Core.norm(imagePoints.get(i), projectedPoints, Core.NORM_L2);
            int n = (int) obj.total();
            errors.add(Math.sqrt(error * error / n));
        }
        
        projectedPoints.release();
        return errors;
    }

    public static int findWorstFrameIndex(List<Double> errors) {
        if (errors == null || errors.isEmpty()) return -1;
        int worstIdx = 0;
        double maxError = errors.get(0);
        for (int i = 1; i < errors.size(); i++) {
            if (errors.get(i) > maxError) {
                maxError = errors.get(i);
                worstIdx = i;
            }
        }
        return worstIdx;
    }
}
