package com.example.visionproject.calibracao.pipeline;

import com.example.visionproject.calibracao.model.CalibrationFrame;
import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.model.PatternSpec;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Pipeline que executa o algoritmo de calibração de Zhang usando OpenCV.
 */
public class ZhangCalibrationPipeline {

    public CalibrationResult run(List<CalibrationFrame> frames, PatternSpec pattern, Size imageSize) {
        if (frames.size() < 15) {
            throw new IllegalStateException("São necessários pelo menos 15 frames para uma calibração confiável.");
        }

        List<Mat> objPoints = new ArrayList<>();
        List<Mat> imgPoints = new ArrayList<>();

        for (CalibrationFrame f : frames) {
            objPoints.add(pattern.generateObjectPoints());
            imgPoints.add(f.getCorners());
        }

        Mat cameraMatrix = new Mat();
        Mat distCoeffs = new Mat();
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();

        long t0 = System.currentTimeMillis();
        // Executa a calibração principal
        double rms = Calib3d.calibrateCamera(
                objPoints,
                imgPoints,
                imageSize,
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs
        );
        long elapsed = System.currentTimeMillis() - t0;

        // Analisa o erro por imagem
        List<Double> perImageErrors = ReprojectionErrorAnalyzer.computePerImage(
                objPoints, imgPoints, rvecs, tvecs, cameraMatrix, distCoeffs
        );

        CalibrationResult result = new CalibrationResult(
                rms, cameraMatrix, distCoeffs, perImageErrors,
                frames.size(), 0, elapsed, new Date(), imageSize
        );

        // Limpeza de recursos temporários
        cameraMatrix.release();
        distCoeffs.release();
        for (Mat m : objPoints) m.release();
        // imgPoints não devem ser liberados aqui pois pertencem aos CalibrationFrames no repositório
        for (Mat m : rvecs) m.release();
        for (Mat m : tvecs) m.release();

        return result;
    }
}
