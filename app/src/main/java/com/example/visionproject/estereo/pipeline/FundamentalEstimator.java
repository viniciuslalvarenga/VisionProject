package com.example.visionproject.estereo.pipeline;

import com.example.visionproject.estereo.model.FundamentalResult;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

public class FundamentalEstimator {
    public static FundamentalResult estimate(List<KeyPoint> kpL, List<KeyPoint> kpR, List<DMatch> good) {
        if (good.size() < 8) {
            throw new IllegalStateException("São necessários pelo menos 8 matches para estimar a matriz fundamental.");
        }

        List<Point> ptsL = new ArrayList<>();
        List<Point> ptsR = new ArrayList<>();
        for (DMatch m : good) {
            // Validar índices antes de acessar
            if (m.queryIdx >= 0 && m.queryIdx < kpL.size() && m.trainIdx >= 0 && m.trainIdx < kpR.size()) {
                ptsL.add(kpL.get(m.queryIdx).pt);
                ptsR.add(kpR.get(m.trainIdx).pt);
            }
        }

        if (ptsL.size() < 8) {
            throw new IllegalStateException("Após validação de índices, matches insuficientes (< 8): " + ptsL.size());
        }

        MatOfPoint2f p1 = new MatOfPoint2f();
        MatOfPoint2f p2 = new MatOfPoint2f();
        try {
            p1.fromList(ptsL);
            p2.fromList(ptsR);

            Mat mask = new Mat();
            Mat F = Calib3d.findFundamentalMat(p1, p2, Calib3d.FM_RANSAC, 3.0, 0.99, mask);

            int inliersCount = Core.countNonZero(mask);

            List<Point> inL = new ArrayList<>();
            List<Point> inR = new ArrayList<>();
            for (int i = 0; i < mask.rows(); i++) {
                if (mask.get(i, 0)[0] != 0) {
                    inL.add(ptsL.get(i));
                    inR.add(ptsR.get(i));
                }
            }

            return new FundamentalResult(F, inliersCount, mask, inL, inR);
        } finally {
            // Liberar MatOfPoint2f
            if (p1 != null) p1.release();
            if (p2 != null) p2.release();
        }
    }
}
