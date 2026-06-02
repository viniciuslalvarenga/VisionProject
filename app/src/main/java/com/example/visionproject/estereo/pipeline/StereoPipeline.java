package com.example.visionproject.estereo.pipeline;

import android.content.Context;

import com.example.visionproject.estereo.export.ImageExporter;
import com.example.visionproject.estereo.factory.SgbmParamsFactory;
import com.example.visionproject.estereo.model.DisparityResult;
import com.example.visionproject.estereo.model.FundamentalResult;
import com.example.visionproject.estereo.model.RectifiedPair;
import com.example.visionproject.estereo.model.SgbmParams;
import com.example.visionproject.estereo.repository.StereoCsvLogger;
import com.example.visionproject.estereo.repository.StereoPairRepository;
import com.example.visionproject.estereo.strategy.OrbFeatureStrategy;
import com.example.visionproject.estereo.strategy.SgbmDisparityStrategy;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

public class StereoPipeline {

    public interface PipelineCallback {
        void onStepDone(String stepName, Mat artifact);
        void onError(String message);
    }

    public void run(Context ctx, File leftFile, File rightFile, Mat K, Mat distCoeffs,
                    String sceneId, int baselineMm, PipelineCallback callback) {
        StereoCsvLogger logger = StereoCsvLogger.getInstance();
        Mat imgL = null, imgR = null, undL = null, undR = null;
        Mat grayL = null, grayR = null, desL = null, desR = null;
        MatOfKeyPoint kpL = null, kpR = null;
        Mat grayRectL = null, grayRectR = null;
        try {
            long t0 = System.currentTimeMillis();
            imgL = Imgcodecs.imread(leftFile.getAbsolutePath());
            imgR = Imgcodecs.imread(rightFile.getAbsolutePath());
            if (imgL.empty() || imgR.empty()) {
                callback.onError("Falha ao carregar par de imagens");
                return;
            }
            undL = new Mat();
            undR = new Mat();
            Calib3d.undistort(imgL, undL, K, distCoeffs);
            Calib3d.undistort(imgR, undR, K, distCoeffs);
            logger.log("UNDISTORT_DONE", "elapsed=" + (System.currentTimeMillis() - t0) + "ms");

            grayL = new Mat();
            grayR = new Mat();
            Imgproc.cvtColor(undL, grayL, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(undR, grayR, Imgproc.COLOR_BGR2GRAY);

            t0 = System.currentTimeMillis();
            OrbFeatureStrategy orb = new OrbFeatureStrategy(2000);
            kpL = new MatOfKeyPoint();
            kpR = new MatOfKeyPoint();
            desL = new Mat();
            desR = new Mat();
            orb.detectAndCompute(grayL, kpL, desL);
            orb.detectAndCompute(grayR, kpR, desR);

            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            List<MatOfDMatch> knn = new ArrayList<>();
            matcher.knnMatch(desL, desR, knn, 2);

            List<DMatch> good = new ArrayList<>();
            for (MatOfDMatch m : knn) {
                DMatch[] arr = m.toArray();
                if (arr.length >= 2 && arr[0].distance < 0.75f * arr[1].distance) {
                    good.add(arr[0]);
                }
            }
            long elapsedOrb = System.currentTimeMillis() - t0;
            logger.logDetailed("ORB_DONE", sceneId, null, baselineMm,
                    knn.size(), good.size(), null, null, null, null,
                    elapsedOrb, null, null, null, null, null, null,
                    "kpL=" + kpL.toList().size() + " kpR=" + kpR.toList().size());

            if (good.size() < 8) {
                callback.onError("Matches insuficientes para F (< 8): " + good.size());
                return;
            }

            Mat matchesArt = new Mat();
            MatOfDMatch goodMat = new MatOfDMatch();
            goodMat.fromList(good);
            Features2d.drawMatches(undL, kpL, undR, kpR, goodMat, matchesArt);
            ImageExporter.saveMat(ctx, matchesArt, "t3_matches_" + sceneId + ".png");
            callback.onStepDone("MATCHES", matchesArt);

            t0 = System.currentTimeMillis();
            FundamentalResult fr = FundamentalEstimator.estimate(kpL.toList(), kpR.toList(), good);
            long elapsedF = System.currentTimeMillis() - t0;
            logger.logDetailed("F_ESTIMATED", sceneId, null, baselineMm,
                    null, null, fr.inliers, null, null, null,
                    elapsedF, null, null, null, null, null, null,
                    "F estimado, inliers=" + fr.inliers);

            Mat epipolarArt = drawEpipolarLines(undL, undR, fr.inliersL, fr.inliersR, fr.F);
            ImageExporter.saveMat(ctx, epipolarArt, "t4_epipolar_" + sceneId + ".png");
            callback.onStepDone("EPIPOLAR", epipolarArt);

            t0 = System.currentTimeMillis();
            RectifiedPair rectified = null;
            try {
                rectified = RectificationEstimator.rectify(undL, undR, fr, K);
            } catch (Exception ex) {
                logger.log("RECTIFY_FAILED", ex.getMessage());
                callback.onError("Retificacao falhou: " + ex.getMessage());
                return;
            }

            long elapsedRect = System.currentTimeMillis() - t0;
            logger.logDetailed("RECTIFY_DONE", sceneId, null, baselineMm,
                    null, null, null, null, null, null,
                    elapsedRect, null, null, null, null, null, null,
                    "stereoRectifyUncalibrated OK");

            Mat rectArt = null;
            Mat grayRectL2 = null, grayRectR2 = null;
            try {
                rectArt = drawHorizontalLines(rectified.rectL, rectified.rectR);
                ImageExporter.saveMat(ctx, rectArt, "t5_rectified_" + sceneId + ".png");
                callback.onStepDone("RECTIFIED", rectArt);

                grayRectL2 = new Mat();
                grayRectR2 = new Mat();
                Imgproc.cvtColor(rectified.rectL, grayRectL2, Imgproc.COLOR_BGR2GRAY);
                Imgproc.cvtColor(rectified.rectR, grayRectR2, Imgproc.COLOR_BGR2GRAY);

                SgbmParams[] presets = {
                        SgbmParamsFactory.fast(),
                        SgbmParamsFactory.balanced(),
                        SgbmParamsFactory.quality()
                };
                String[] presetNames = {"FAST", "BALANCED", "QUALITY"};
                Mat bestDispColor = null;
                Mat bestDisp32f = null;
                double bestPct = -1.0;
                String bestPreset = null;

                for (int i = 0; i < presets.length; i++) {
                    SgbmDisparityStrategy sgbm = new SgbmDisparityStrategy(presets[i]);
                    DisparityResult dr = sgbm.compute(grayRectL2, grayRectR2);
                    double pctValid = computePctValid(dr.disp32f);
                    logger.logDetailed("SGBM_RUN", sceneId, null, baselineMm,
                            null, null, null, presetNames[i],
                            presets[i].numDisparities, presets[i].blockSize,
                            dr.elapsedMs, pctValid, null, null, null, null, null,
                            "P1=" + presets[i].P1 + " P2=" + presets[i].P2);

                    Mat dispColor = disparityColorMap(dr.disp32f);
                    ImageExporter.saveMat(ctx, dispColor, "t6_disp_" + presetNames[i] + "_" + sceneId + ".png");
                    if (pctValid > bestPct) {
                        if (bestDispColor != null) bestDispColor.release();
                        if (bestDisp32f != null) bestDisp32f.release();
                        bestPct = pctValid;
                        bestDispColor = dispColor;
                        bestDisp32f = dr.disp32f.clone();
                        bestPreset = presetNames[i];
                    } else {
                        dispColor.release();
                    }
                    dr.disp32f.release();
                }

                if (bestDispColor != null && bestDisp32f != null) {
                    callback.onStepDone("DISPARITY", bestDispColor);
                    Mat disp8 = new Mat();
                    Core.normalize(bestDisp32f, disp8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
                    Mat disp8filt = new Mat();
                    Imgproc.medianBlur(disp8, disp8filt, 5);

                    Mat dispColorFilt = new Mat();
                    Imgproc.applyColorMap(disp8filt, dispColorFilt, Imgproc.COLORMAP_TURBO);
                    Mat sideBySide = new Mat();
                    Core.hconcat(Arrays.asList(bestDispColor, dispColorFilt), sideBySide);
                    ImageExporter.saveMat(ctx, sideBySide, "t7_postprocess_" + sceneId + ".png");
                    callback.onStepDone("POSTPROCESS", sideBySide);

                    double pctBefore = computePctValid(bestDisp32f);
                    Mat disp32filt = new Mat();
                    disp8filt.convertTo(disp32filt, CvType.CV_32F);
                    double pctAfter = computePctValid(disp32filt);
                    logger.logDetailed("POSTPROCESS_DONE", sceneId, null, baselineMm,
                            null, null, null, null, null, null, null,
                            pctAfter, null, null, null, null, null,
                            "best_preset=" + bestPreset + " pct_before=" + String.format(Locale.US, "%.4f", pctBefore));

                    StereoPairRepository.getInstance().setLastDisparity(bestDisp32f.clone());

                    disp8.release();
                    disp8filt.release();
                    dispColorFilt.release();
                    sideBySide.release();
                    disp32filt.release();
                    bestDispColor.release();
                    bestDisp32f.release();
                }
            } finally {
                // Liberar RectifiedPair
                if (rectified != null) {
                    if (rectified.rectL != null) rectified.rectL.release();
                    if (rectified.rectR != null) rectified.rectR.release();
                    if (rectified.H1 != null) rectified.H1.release();
                    if (rectified.H2 != null) rectified.H2.release();
                }
                if (rectArt != null) rectArt.release();
                if (grayRectL2 != null) grayRectL2.release();
                if (grayRectR2 != null) grayRectR2.release();
            }

        } catch (Exception e) {
            callback.onError("Pipeline error: " + e.getMessage());
            logger.log("PIPELINE_ERROR", e.getMessage() == null ? "(sem mensagem)" : e.getMessage());
        } finally {
            if (imgL != null) imgL.release();
            if (imgR != null) imgR.release();
            if (undL != null) undL.release();
            if (undR != null) undR.release();
            if (grayL != null) grayL.release();
            if (grayR != null) grayR.release();
            if (desL != null) desL.release();
            if (desR != null) desR.release();
            if (grayRectL != null) grayRectL.release();
            if (grayRectR != null) grayRectR.release();
        }
    }

    private Mat drawEpipolarLines(Mat undL, Mat undR, List<Point> inL, List<Point> inR, Mat F) {
        int N = Math.min(10, inL.size());
        Random rand = new Random(42);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < N; i++) indices.add(rand.nextInt(inL.size()));
        Scalar[] colors = {
                new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255),
                new Scalar(255, 255, 0), new Scalar(255, 0, 255), new Scalar(0, 255, 255),
                new Scalar(128, 0, 255), new Scalar(255, 128, 0), new Scalar(0, 128, 255),
                new Scalar(128, 255, 0)
        };
        List<Point> selL = new ArrayList<>(), selR = new ArrayList<>();
        for (int idx : indices) { selL.add(inL.get(idx)); selR.add(inR.get(idx)); }
        MatOfPoint2f p1 = new MatOfPoint2f();
        p1.fromList(selL);
        Mat lines2 = new Mat();
        Calib3d.computeCorrespondEpilines(p1, 1, F, lines2);

        Mat artL = undL.clone();
        Mat artR = undR.clone();
        int w = artR.cols();
        for (int i = 0; i < N; i++) {
            Scalar c = colors[i % colors.length];
            double[] line = lines2.get(i, 0);
            double a = line[0], b = line[1], cc = line[2];
            if (Math.abs(b) > 1e-6) {
                Point pt1 = new Point(0, -cc / b);
                Point pt2 = new Point(w - 1, (-cc - a * (w - 1)) / b);
                Imgproc.line(artR, pt1, pt2, c, 1);
            }
            Imgproc.circle(artL, selL.get(i), 6, c, 2);
            Imgproc.circle(artR, selR.get(i), 6, c, 2);
        }
        Mat combined = new Mat();
        Core.hconcat(Arrays.asList(artL, artR), combined);
        artL.release(); artR.release(); lines2.release();
        return combined;
    }

    private Mat drawHorizontalLines(Mat rectL, Mat rectR) {
        Mat artL = rectL.clone();
        Mat artR = rectR.clone();
        int h = artL.rows();
        Scalar[] colors = {
                new Scalar(0, 255, 255), new Scalar(255, 255, 0), new Scalar(255, 0, 255),
                new Scalar(0, 255, 0), new Scalar(255, 128, 0), new Scalar(0, 128, 255),
                new Scalar(255, 0, 128), new Scalar(128, 255, 128), new Scalar(255, 255, 255),
                new Scalar(0, 0, 255)
        };
        for (int i = 0; i < 10; i++) {
            int y = (h * (i + 1)) / 11;
            Scalar c = colors[i % colors.length];
            Imgproc.line(artL, new Point(0, y), new Point(artL.cols() - 1, y), c, 1);
            Imgproc.line(artR, new Point(0, y), new Point(artR.cols() - 1, y), c, 1);
        }
        Mat combined = new Mat();
        Core.hconcat(Arrays.asList(artL, artR), combined);
        artL.release(); artR.release();
        return combined;
    }

    private Mat disparityColorMap(Mat disp32f) {
        Mat disp8 = new Mat();
        Core.normalize(disp32f, disp8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        Mat colored = new Mat();
        Imgproc.applyColorMap(disp8, colored, Imgproc.COLORMAP_TURBO);
        disp8.release();
        return colored;
    }

    private double computePctValid(Mat disp32f) {
        Mat mask = new Mat();
        Core.compare(disp32f, new Scalar(0.5), mask, Core.CMP_GT);
        int valid = Core.countNonZero(mask);
        int total = (int) (disp32f.total());
        mask.release();
        return total > 0 ? (double) valid / total : 0;
    }
}
