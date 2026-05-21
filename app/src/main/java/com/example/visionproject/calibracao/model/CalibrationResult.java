package com.example.visionproject.calibracao.model;

import com.example.visionproject.modelocamera.model.DistortionCoefficients;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Resultado imutável da calibração.
 */
public class CalibrationResult {
    private final double rms;
    private final Mat cameraMatrix;
    private final Mat distCoeffs;
    private final List<Double> perImageReprojectionError;
    private final int imagesUsed;
    private final int imagesRejected;
    private final long elapsedMsTotal;
    private final Date calibrationDateTime;
    private final Size imageSize;

    public CalibrationResult(double rms, Mat cameraMatrix, Mat distCoeffs, List<Double> perImageReprojectionError,
                             int imagesUsed, int imagesRejected, long elapsedMsTotal, Date calibrationDateTime, Size imageSize) {
        this.rms = rms;
        this.cameraMatrix = cameraMatrix.clone();
        this.distCoeffs = distCoeffs.clone();
        this.perImageReprojectionError = perImageReprojectionError;
        this.imagesUsed = imagesUsed;
        this.imagesRejected = imagesRejected;
        this.elapsedMsTotal = elapsedMsTotal;
        this.calibrationDateTime = calibrationDateTime;
        this.imageSize = imageSize;
    }

    public double getRms() { return rms; }
    public Mat getCameraMatrix() { return cameraMatrix; }
    public Mat getDistCoeffs() { return distCoeffs; }
    public List<Double> getPerImageReprojectionError() { return perImageReprojectionError; }
    public int getImagesUsed() { return imagesUsed; }
    public long getElapsedMsTotal() { return elapsedMsTotal; }
    public Size getImageSize() { return imageSize; }

    public double getFx() { return cameraMatrix.get(0, 0)[0]; }
    public double getFy() { return cameraMatrix.get(1, 1)[0]; }
    public double getCx() { return cameraMatrix.get(0, 2)[0]; }
    public double getCy() { return cameraMatrix.get(1, 2)[0]; }
    public double getSkew() { return cameraMatrix.get(0, 1)[0]; }

    public double getK1() { return distCoeffs.total() > 0 ? distCoeffs.get(0, 0)[0] : 0.0; }
    public double getK2() { return distCoeffs.total() > 1 ? distCoeffs.get(0, 1)[0] : 0.0; }
    public double getP1() { return distCoeffs.total() > 2 ? distCoeffs.get(0, 2)[0] : 0.0; }
    public double getP2() { return distCoeffs.total() > 3 ? distCoeffs.get(0, 3)[0] : 0.0; }
    public double getK3() { return distCoeffs.total() > 4 ? distCoeffs.get(0, 4)[0] : 0.0; }

    public DistortionCoefficients.DistortionType getDistortionType() {
        double k1 = getK1();
        if (k1 < -0.05) return DistortionCoefficients.DistortionType.BARREL;
        if (k1 > 0.05) return DistortionCoefficients.DistortionType.PINCUSHION;
        return DistortionCoefficients.DistortionType.NONE;
    }

    public double[] getDistortionCoefficients() {
        double[] d = new double[(int) distCoeffs.total()];
        for (int i = 0; i < d.length; i++) {
            d[i] = distCoeffs.get(0, i)[0];
        }
        return d;
    }

    public JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("version", "1.0");
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);
        json.put("calibration_date", sdf.format(calibrationDateTime));
        
        JSONObject imgSizeJson = new JSONObject();
        imgSizeJson.put("width", imageSize.width);
        imgSizeJson.put("height", imageSize.height);
        json.put("image_size", imgSizeJson);

        JSONObject intrinsics = new JSONObject();
        intrinsics.put("fx", getFx());
        intrinsics.put("fy", getFy());
        intrinsics.put("cx", getCx());
        intrinsics.put("cy", getCy());
        intrinsics.put("skew", getSkew());
        json.put("intrinsics", intrinsics);

        JSONObject distortion = new JSONObject();
        distortion.put("k1", getK1());
        distortion.put("k2", getK2());
        distortion.put("p1", getP1());
        distortion.put("p2", getP2());
        distortion.put("k3", getK3());
        distortion.put("type", getDistortionType().name());
        json.put("distortion", distortion);

        json.put("rms_reprojection_error_px", rms);
        json.put("images_used", imagesUsed);
        json.put("images_rejected", imagesRejected);
        json.put("elapsed_ms_total", elapsedMsTotal);

        return json;
    }

    public void release() {
        if (cameraMatrix != null) cameraMatrix.release();
        if (distCoeffs != null) distCoeffs.release();
    }
}
