package com.example.visionproject.estereo.viewmodel;

import android.app.Application;
import android.graphics.Bitmap;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.estereo.model.StereoPair;
import com.example.visionproject.estereo.pipeline.DepthTriangulator;
import com.example.visionproject.estereo.repository.StereoCsvLogger;
import com.example.visionproject.estereo.repository.StereoPairRepository;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class DepthViewModel extends AndroidViewModel {
    private final MutableLiveData<Bitmap> disparityBitmap = new MutableLiveData<>();
    private final MutableLiveData<String> depthInfo = new MutableLiveData<>("Toque no mapa para medir");
    private Mat currentDepthMap;
    private Mat lastDisparity;
    private Double lastMeasuredZ = null;
    private Integer lastClickedX = null, lastClickedY = null;

    public DepthViewModel(@NonNull Application application) {
        super(application);
    }

    public LiveData<Bitmap> getDisparityBitmap() { return disparityBitmap; }
    public LiveData<String> getDepthInfo() { return depthInfo; }

    public void init() {
        Mat disparity = StereoPairRepository.getInstance().getLastDisparity();
        if (disparity != null) {
            this.lastDisparity = disparity.clone();
            calculateDepth();
            updateDisparityPreview();
        } else {
            depthInfo.setValue("Erro: Mapa de disparidade nao disponivel.");
        }
    }

    public void exportPly() {
        if (currentDepthMap == null) {
            depthInfo.postValue("Nenhuma profundidade calculada para exportar.");
            return;
        }
        CalibrationResult calib = CalibrationJsonStore.load(getApplication());
        if (calib == null) return;

        double fx = calib.getCameraMatrix().get(0, 0)[0];
        double cx = calib.getCameraMatrix().get(0, 2)[0];
        double cy = calib.getCameraMatrix().get(1, 2)[0];

        java.io.File dir = new java.io.File(
                android.os.Environment.getExternalStoragePublicDirectory(
                        android.os.Environment.DIRECTORY_DOCUMENTS), "VisionProject");
        if (!dir.exists()) dir.mkdirs();
        java.io.File plyFile = new java.io.File(dir, "point_cloud_" + System.currentTimeMillis() + ".ply");

        try {
            com.example.visionproject.estereo.export.PlyExporter.export(currentDepthMap, fx, cx, cy, plyFile);
            depthInfo.postValue("Exportado: " + plyFile.getName());
            StereoCsvLogger.getInstance().logDetailed("PLY_EXPORTED", null, null, null,
                    null, null, null, null, null, null, null, null, null,
                    null, null, null, plyFile.getAbsolutePath(), null);
        } catch (Exception e) {
            depthInfo.postValue("Erro ao exportar: " + e.getMessage());
        }
    }

    private void calculateDepth() {
        CalibrationResult calib = CalibrationJsonStore.load(getApplication());
        StereoPair pair = StereoPairRepository.getInstance().getCurrentPair();
        if (calib != null && pair != null && lastDisparity != null) {
            double fx = calib.getCameraMatrix().get(0, 0)[0];
            double baselineMeters = pair.baselineMm / 1000.0;
            currentDepthMap = DepthTriangulator.triangulate(lastDisparity, fx, baselineMeters);
        }
    }

    private void updateDisparityPreview() {
        if (lastDisparity == null) return;
        Mat disp8 = new Mat();
        Core.normalize(lastDisparity, disp8, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        Mat colored = new Mat();
        Imgproc.applyColorMap(disp8, colored, Imgproc.COLORMAP_TURBO);
        Bitmap bmp = Bitmap.createBitmap(colored.cols(), colored.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(colored, bmp);
        disparityBitmap.postValue(bmp);
        disp8.release();
        colored.release();
    }

    public void onMapClick(int x, int y) {
        if (currentDepthMap == null) return;
        if (x < 0 || y < 0 || x >= currentDepthMap.cols() || y >= currentDepthMap.rows()) return;

        int radius = 6;
        int minValid = 10;
        int startX = Math.max(0, x - radius);
        int endX = Math.min(currentDepthMap.cols() - 1, x + radius);
        int startY = Math.max(0, y - radius);
        int endY = Math.min(currentDepthMap.rows() - 1, y + radius);

        java.util.List<Float> validDepths = new java.util.ArrayList<>();
        for (int yy = startY; yy <= endY; yy++) {
            for (int xx = startX; xx <= endX; xx++) {
                double[] z = currentDepthMap.get(yy, xx);
                if (z != null && z[0] > 0.001) {
                    validDepths.add((float) z[0]);
                }
            }
        }

        if (validDepths.size() < minValid) {
            lastMeasuredZ = null;
            depthInfo.setValue("Ponto sem profundidade confiavel na regiao selecionada.");
            return;
        }

        java.util.Collections.sort(validDepths);
        double medianZ = validDepths.get(validDepths.size() / 2);
        lastMeasuredZ = medianZ;
        lastClickedX = x;
        lastClickedY = y;
        depthInfo.setValue(String.format(java.util.Locale.US,
                "Pixel (%d, %d) -> Z = %.3f m (%d pixels)", x, y, medianZ, validDepths.size()));
        StereoCsvLogger.getInstance().logDetailed("DEPTH_QUERY", null, null, null,
                null, null, null, null, null, null, null, null, null,
                null, medianZ, null,
                "pixel=(" + x + "," + y + ") count=" + validDepths.size(), null);
    }

    /** T8 - compara o ultimo Z medido com distancia real (fita metrica). */
    public void compareWithReference(double referenceMeters) {
        if (lastMeasuredZ == null) {
            depthInfo.postValue("Toque primeiro em um ponto do mapa para medir Z.");
            return;
        }
        if (referenceMeters <= 0) {
            depthInfo.postValue("Distancia de referencia invalida.");
            return;
        }
        double erroAbs = Math.abs(lastMeasuredZ - referenceMeters);
        double erroPct = erroAbs / referenceMeters * 100.0;
        String msg = String.format(java.util.Locale.US,
                "Medido: %.3f m | Real: %.3f m | Erro: %.1f%% (%.3f m)",
                lastMeasuredZ, referenceMeters, erroPct, erroAbs);
        depthInfo.postValue(msg);
        StereoCsvLogger.getInstance().logDetailed("REFERENCE_COMPARED", null, null, null,
                null, null, null, null, null, null, null, null, null,
                referenceMeters, lastMeasuredZ, erroPct,
                "pixel=(" + lastClickedX + "," + lastClickedY + ")", null);
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        if (currentDepthMap != null) currentDepthMap.release();
        if (lastDisparity != null) lastDisparity.release();
    }
}
