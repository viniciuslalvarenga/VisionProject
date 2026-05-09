package com.example.visionproject.modelocamera.strategy;

import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

/**
 * Estratégia para corrigir a distorção de um único ponto.
 */
public final class PointUndistortStrategy implements UndistortStrategy<Point, Point> {

    @Override
    public Point undistort(Point input, CameraIntrinsics k, DistortionCoefficients d) {
        MatOfPoint2f distortedPts = new MatOfPoint2f(input);
        MatOfPoint2f undistortedPts = new MatOfPoint2f();

        // undistortPoints retorna coordenadas normalizadas se P ou R não forem passados.
        Calib3d.undistortPoints(distortedPts, undistortedPts, k.toMat(), d.toMat());

        Point p = undistortedPts.toArray()[0];

        // Para voltar para pixels: x' = x*fx + cx, y' = y*fy + cy
        double correctedX = p.x * k.getFx() + k.getCx();
        double correctedY = p.y * k.getFy() + k.getCy();

        distortedPts.release();
        undistortedPts.release();
        
        return new Point(correctedX, correctedY);
    }
}
