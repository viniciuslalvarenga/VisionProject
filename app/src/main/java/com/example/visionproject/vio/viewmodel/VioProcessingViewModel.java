package com.example.visionproject.vio.viewmodel;

import android.app.Application;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.estereo.factory.SgbmParamsFactory;
import com.example.visionproject.estereo.model.DisparityResult;
import com.example.visionproject.estereo.model.SgbmParams;
import com.example.visionproject.estereo.strategy.SgbmDisparityStrategy;
import com.example.visionproject.vio.VioState;
import com.example.visionproject.vio.export.PlyExporterMetric;
import com.example.visionproject.vio.model.CalibratedRectifyResult;
import com.example.visionproject.vio.model.DepthEstimateResult;
import com.example.visionproject.vio.model.KeyframePair;
import com.example.visionproject.vio.model.RelativePoseResult;
import com.example.visionproject.vio.pipeline.CalibratedRectifier;
import com.example.visionproject.vio.pipeline.DepthFromQ;
import com.example.visionproject.vio.pipeline.ImuNavigator;
import com.example.visionproject.vio.pipeline.RelativePoseEstimator;
import com.example.visionproject.vio.pipeline.VioFusion;
import com.example.visionproject.vio.repository.KeyframePairRepository;
import com.example.visionproject.vio.repository.VioCsvLogger;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VioProcessingViewModel extends AndroidViewModel {
    private final MutableLiveData<VioState>  state   = new MutableLiveData<>(VioState.IDLE);
    private final MutableLiveData<Bitmap>    artifact = new MutableLiveData<>();
    private final MutableLiveData<String>    status  = new MutableLiveData<>("");
    private final MutableLiveData<DepthEstimateResult> depthResult = new MutableLiveData<>();

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private Mat xyzMat;
    private String currentSceneId;
    private RelativePoseResult poseResult;
    private CalibratedRectifyResult rectResult;

    public VioProcessingViewModel(@NonNull Application app) { super(app); }

    public LiveData<VioState>            getState()      { return state; }
    public LiveData<Bitmap>             getArtifact()   { return artifact; }
    public LiveData<String>             getStatus()     { return status; }
    public LiveData<DepthEstimateResult> getDepthResult(){ return depthResult; }
    public Mat getXyzMat()                               { return xyzMat; }

    public void runPipeline() {
        KeyframePair pair = KeyframePairRepository.getInstance().getPair();
        if (pair == null) { state.postValue(VioState.ERROR); status.postValue("Par não encontrado."); return; }

        CalibrationResult cr = CalibrationJsonStore.load(getApplication());
        if (cr == null) { state.postValue(VioState.ERROR); status.postValue("Calibração não encontrada."); return; }

        state.postValue(VioState.PROCESSING);
        currentSceneId = "vio_" + System.currentTimeMillis();

        executor.execute(() -> {
            VioCsvLogger logger = VioCsvLogger.getInstance();
            Mat K = cr.getCameraMatrix();
            Mat D = cr.getDistCoeffs();

            try {
                // Step 1: ORB matches visualization
                status.postValue("ORB matches...");
                Mat matchViz = new Mat();
                org.opencv.core.MatOfDMatch goodMat = new org.opencv.core.MatOfDMatch();
                goodMat.fromList(pair.matches.good);
                Features2d.drawMatches(pair.matches.undL, pair.matches.kpL,
                        pair.matches.undR, pair.matches.kpR, goodMat, matchViz);
                postBitmap(matchViz);
                logger.logDetailed("ORB_DONE", currentSceneId, null, null,
                        null,null,null,null,null,null,null,null,null,null,
                        pair.baseline_m, pair.parallax_px, pair.matches.good.size(),
                        null,null,null,null,null,null,null,"matches vizualized");
                matchViz.release();

                // Step 2: Essential matrix + pose
                status.postValue("Estimando pose relativa (E + RANSAC)...");
                RelativePoseEstimator estimator = new RelativePoseEstimator();
                poseResult = estimator.estimate(pair, K, D);
                logger.logDetailed("POSE_RECOVERED", currentSceneId, null, null,
                        null,null,null,null,null,null,null,null,null,null,
                        poseResult.scale_m, null, null,
                        poseResult.inliers, poseResult.rotAngleDeg,
                        cr.getFx(), null, null, null, null,
                        String.format(Locale.US, "inliers=%d angle=%.1fdeg scale=%.3fm",
                                poseResult.inliers, poseResult.rotAngleDeg, poseResult.scale_m));
                if (poseResult.inliers < 50) {
                    status.postValue("Aviso: inliers < 50 (" + poseResult.inliers + "). Resultado pode ser impreciso.");
                }
                if (poseResult.rotAngleDeg > 15.0) {
                    logger.log("POSE_DEGENERATE", "rotAngleDeg=" + poseResult.rotAngleDeg);
                }

                // Step 3: Rectification
                status.postValue("Retificando par (stereoRectify calibrado)...");
                Size imgSize = new Size(pair.a.rgb.cols(), pair.a.rgb.rows());
                CalibratedRectifier rectifier = new CalibratedRectifier();
                rectResult = rectifier.rectify(pair.matches.undL, pair.matches.undR,
                        K, D, poseResult.R_rel, poseResult.t_metric, imgSize);
                Mat rectViz = drawHLines(rectResult.rectL, rectResult.rectR);
                postBitmap(rectViz);
                rectViz.release();
                logger.log("RECTIFY_CALIBRATED_DONE", currentSceneId);

                // Step 4: SGBM x3 presets
                status.postValue("Computando disparidade SGBM (3 presets)...");
                Mat gL = new Mat(), gR = new Mat();
                Imgproc.cvtColor(rectResult.rectL, gL, Imgproc.COLOR_BGR2GRAY);
                Imgproc.cvtColor(rectResult.rectR, gR, Imgproc.COLOR_BGR2GRAY);

                SgbmParams[] presets = {SgbmParamsFactory.fast(), SgbmParamsFactory.balanced(), SgbmParamsFactory.quality()};
                String[] names = {"FAST","BALANCED","QUALITY"};
                Mat bestDisp32f = null; double bestPct = -1;
                for (int i = 0; i < 3; i++) {
                    SgbmDisparityStrategy sgbm = new SgbmDisparityStrategy(presets[i]);
                    DisparityResult dr = sgbm.compute(gL, gR);
                    double pct = pctValid(dr.disp32f);
                    logger.logDetailed("SGBM_RUN", currentSceneId, null, null,
                            null,null,null,null,null,null,null,null,null,null,
                            null,null,null,null,null,null,null,null,null,null,
                            "preset="+names[i]+" pct="+String.format(Locale.US,"%.3f",pct));
                    Mat colored = colormap(dr.disp32f);
                    postBitmap(colored);
                    colored.release();
                    if (pct > bestPct) {
                        if (bestDisp32f != null) bestDisp32f.release();
                        bestPct = pct;
                        bestDisp32f = dr.disp32f.clone();
                    }
                    dr.disp32f.release();
                }
                gL.release(); gR.release();

                // Step 5: Depth via Q
                status.postValue("Reprojetando para 3D (Q matrix)...");
                if (bestDisp32f != null) {
                    if (xyzMat != null) xyzMat.release();
                    xyzMat = DepthFromQ.reproject(bestDisp32f, rectResult.Q);
                    double mZ = DepthFromQ.medianZ(xyzMat);
                    depthResult.postValue(new DepthEstimateResult(xyzMat, mZ, Double.NaN));
                    logger.logDetailed("DEPTH_QUERY", currentSceneId, null, null,
                            null,null,null,null,null,null,null,null,null,null,
                            null,null,null,null,null,cr.getFx(),mZ,null,null,null,
                            "median_Z="+String.format(Locale.US,"%.3f",mZ)+"m");
                    // Show colormap of best disparity as final artifact
                    Mat dispColor = colormap(bestDisp32f);
                    postBitmap(dispColor);
                    dispColor.release();
                    bestDisp32f.release();
                }

                // Step 6: VIO Fusion
                VioFusion fusion = new VioFusion();
                double[] visualP = {0, 0, poseResult.scale_m};
                float[]  visualQ = poseResult.R_rel.rows() > 0
                        ? matToQuat(poseResult.R_rel) : new float[]{0,0,0,1};
                com.example.visionproject.vio.model.Pose imuPose =
                        ImuNavigator.getInstance().getPoseAt(pair.b.tNs);
                fusion.fusePose(visualP, visualQ, imuPose.p, imuPose.q);
                double[] fp = fusion.getFusedPosition();
                logger.logDetailed("VIO_FUSION_STEP", currentSceneId, null, null,
                        fp[0],fp[1],fp[2],null,null,null,null,null,null,null,
                        null,null,null,null,null,null,null,null,null,null,
                        "alpha="+fusion.getAlpha());

                state.postValue(VioState.DONE);
                status.postValue("Pipeline concluído!");

            } catch (Exception e) {
                logger.log("PIPELINE_ERROR", e.getMessage());
                state.postValue(VioState.ERROR);
                status.postValue("Erro: " + e.getMessage());
            }
        });
    }

    public void logScaleError(double zEst, double zReal) {
        double err = Math.abs(zEst - zReal) / zReal * 100.0;
        VioCsvLogger.getInstance().logDetailed("SCALE_ERROR_ANALYSIS",
                currentSceneId, null, null,
                null,null,null,null,null,null,null,null,null,null,
                null,null,null,null,null,null,zEst,zReal,err,null,
                String.format(Locale.US,"Z_est=%.3f Z_real=%.3f err=%.1f%%",zEst,zReal,err));
    }

    public String exportPly() {
        if (xyzMat == null) return null;
        String name = PlyExporterMetric.export(getApplication(), xyzMat, currentSceneId != null ? currentSceneId : "vio");
        if (name != null) VioCsvLogger.getInstance().log("PLY_EXPORTED_METRIC", name);
        return name;
    }

    private void postBitmap(Mat m) {
        if (m == null || m.empty()) return;
        try {
            Bitmap bmp = Bitmap.createBitmap(m.cols(), m.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(m, bmp);
            artifact.postValue(bmp);
        } catch (Exception ignored) {}
    }

    private static Mat drawHLines(Mat l, Mat r) {
        Mat aL = l.clone(), aR = r.clone();
        int h = aL.rows();
        for (int i = 1; i <= 10; i++) {
            int y = h * i / 11;
            org.opencv.core.Scalar c = new org.opencv.core.Scalar(0, 255, 255);
            Imgproc.line(aL, new org.opencv.core.Point(0,y), new org.opencv.core.Point(aL.cols()-1,y), c, 1);
            Imgproc.line(aR, new org.opencv.core.Point(0,y), new org.opencv.core.Point(aR.cols()-1,y), c, 1);
        }
        Mat out = new Mat();
        Core.hconcat(Arrays.asList(aL, aR), out);
        aL.release(); aR.release();
        return out;
    }

    private static Mat colormap(Mat disp32f) {
        Mat d8 = new Mat();
        Core.normalize(disp32f, d8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        Mat colored = new Mat();
        Imgproc.applyColorMap(d8, colored, Imgproc.COLORMAP_TURBO);
        d8.release();
        return colored;
    }

    private static double pctValid(Mat d32f) {
        Mat mask = new Mat();
        Core.compare(d32f, new org.opencv.core.Scalar(0.5), mask, Core.CMP_GT);
        int valid = Core.countNonZero(mask);
        mask.release();
        return d32f.total() > 0 ? (double) valid / d32f.total() : 0;
    }

    private static float[] matToQuat(Mat R) {
        double tr = R.get(0,0)[0] + R.get(1,1)[0] + R.get(2,2)[0];
        float w = (float) Math.sqrt(Math.max(0, 1 + tr)) / 2;
        float x = (float) Math.sqrt(Math.max(0, 1 + R.get(0,0)[0] - R.get(1,1)[0] - R.get(2,2)[0])) / 2;
        float y = (float) Math.sqrt(Math.max(0, 1 - R.get(0,0)[0] + R.get(1,1)[0] - R.get(2,2)[0])) / 2;
        float z = (float) Math.sqrt(Math.max(0, 1 - R.get(0,0)[0] - R.get(1,1)[0] + R.get(2,2)[0])) / 2;
        x = R.get(2,1)[0] - R.get(1,2)[0] < 0 ? -x : x;
        y = R.get(0,2)[0] - R.get(2,0)[0] < 0 ? -y : y;
        z = R.get(1,0)[0] - R.get(0,1)[0] < 0 ? -z : z;
        return new float[]{x, y, z, w};
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        executor.shutdown();
        if (xyzMat != null) xyzMat.release();
    }
}
