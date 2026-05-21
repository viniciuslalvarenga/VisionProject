package com.example.visionproject.calibracao.repository;

import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import com.example.visionproject.calibracao.model.CalibrationFrame;
import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.shared.DeviceInfoProvider;

import java.io.OutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Logger CSV para o Módulo 3, seguindo a especificação de rastreabilidade total.
 * v2: Escrita em background thread e novos eventos.
 */
public class CalibrationCsvLogger {
    private static final String TAG = "CalibrationCsvLogger";
    private static final String HEADER = "session_id,timestamp_ms,timestamp_iso,event_type," +
            "frame_index,corners_found,blur_score,pose_stability_score,coverage_region," +
            "fx,fy,cx,cy,k1,k2,p1,p2,k3,distortion_type," +
            "rms_reprojection_px,per_image_error_px," +
            "image_filename,image_w,image_h," +
            "elapsed_ms,images_used,images_rejected," +
            "rejection_reason,json_path," +
            "device_model,android_version,notes\n";

    private final String sessionId;
    private final SimpleDateFormat isoFormat;
    private final StringBuilder logBuffer = new StringBuilder();
    private final ExecutorService io = Executors.newSingleThreadExecutor();

    private CalibrationCsvLogger() {
        this.sessionId = "CAL_" + System.currentTimeMillis();
        this.isoFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        this.isoFormat.setTimeZone(TimeZone.getTimeZone("UTC"));
        synchronized (logBuffer) {
            logBuffer.append(HEADER);
        }
        logEvent("SESSION_START", "App started: " + DeviceInfoProvider.getDeviceModel());
    }

    private static class Holder {
        static final CalibrationCsvLogger INSTANCE = new CalibrationCsvLogger();
    }

    public static CalibrationCsvLogger getInstance() {
        return Holder.INSTANCE;
    }

    public void logEvent(String eventType, String notes) {
        appendLine(eventType, -1, -1, -1, -1, null, -1, -1, notes);
    }

    public void logFrameCaptured(CalibrationFrame frame, int index) {
        appendLine("FRAME_CAPTURED", index, frame.getCornersFound(), frame.getBlurScore(), frame.getPoseStabilityScore(), null, -1, -1, "");
    }

    public void logFrameDetected(int corners, double blur) {
        appendLine("FRAME_DETECTED", -1, corners, blur, -1, null, -1, -1, "");
    }

    public void logFrameRejected(String reason, double blur) {
        appendLine("FRAME_REJECTED", -1, -1, blur, -1, null, -1, -1, reason);
    }

    public void logCalibrationStarted(int images) {
        appendLine("CALIBRATION_STARTED", -1, -1, -1, -1, null, -1, -1, "Starting with " + images + " frames");
    }

    public void logJsonExported(String path) {
        appendLine("JSON_EXPORTED", -1, -1, -1, -1, null, -1, -1, path);
    }

    public void logUndistortPreview(int w, int h, long elapsed) {
        appendLine("UNDISTORT_PREVIEW", -1, -1, -1, -1, null, elapsed, -1, String.format(Locale.US, "Size: %dx%d", w, h));
    }

    public void logCalibrationDone(CalibrationResult res, List<Double> perImageErrors) {
        long now = System.currentTimeMillis();
        String iso = isoFormat.format(new Date(now));
        
        io.execute(() -> {
            String doneLine = formatLine("CALIBRATION_DONE", -1, -1, -1, -1, res, res.getElapsedMsTotal(), -1, "", now, iso);
            synchronized (logBuffer) {
                logBuffer.append(doneLine).append("\n");
                if (perImageErrors != null) {
                    for (int i = 0; i < perImageErrors.size(); i++) {
                        String line = formatLine("PER_IMAGE_ERROR", i, -1, -1, -1, null, -1, perImageErrors.get(i), "", now, iso);
                        logBuffer.append(line).append("\n");
                    }
                }
            }
        });
    }

    private void appendLine(String eventType, int frameIdx, int corners, double blur, double stability, 
                            CalibrationResult res, long elapsed, double perImageError, String notes) {
        long now = System.currentTimeMillis();
        String iso = isoFormat.format(new Date(now));
        io.execute(() -> {
            String line = formatLine(eventType, frameIdx, corners, blur, stability, res, elapsed, perImageError, notes, now, iso);
            synchronized (logBuffer) {
                logBuffer.append(line).append("\n");
            }
        });
    }

    private String formatLine(String eventType, int frameIdx, int corners, double blur, double stability, 
                             CalibrationResult res, long elapsed, double perImageError, String notes,
                             long now, String iso) {
        String[] fields = new String[32];
        for (int i = 0; i < 32; i++) fields[i] = "";
        
        fields[0] = sessionId;
        fields[1] = String.valueOf(now);
        fields[2] = iso;
        fields[3] = eventType;
        if (frameIdx >= 0) fields[4] = String.valueOf(frameIdx);
        if (corners >= 0) fields[5] = String.valueOf(corners);
        if (blur >= 0) fields[6] = String.format(Locale.US, "%.2f", blur);
        if (stability >= 0) fields[7] = String.format(Locale.US, "%.2f", stability);
        
        if (res != null) {
            fields[9] = String.format(Locale.US, "%.4f", res.getFx());
            fields[10] = String.format(Locale.US, "%.4f", res.getFy());
            fields[11] = String.format(Locale.US, "%.4f", res.getCx());
            fields[12] = String.format(Locale.US, "%.4f", res.getCy());
            fields[13] = String.format(Locale.US, "%.6f", res.getK1());
            fields[14] = String.format(Locale.US, "%.6f", res.getK2());
            fields[15] = String.format(Locale.US, "%.6f", res.getP1());
            fields[16] = String.format(Locale.US, "%.6f", res.getP2());
            fields[17] = String.format(Locale.US, "%.6f", res.getK3());
            fields[18] = res.getDistortionType().name();
            fields[19] = String.format(Locale.US, "%.4f", res.getRms());
            fields[25] = String.valueOf(res.getImagesUsed());
        }
        
        if (perImageError >= 0) fields[20] = String.format(Locale.US, "%.4f", perImageError);
        if (elapsed > 0) fields[24] = String.valueOf(elapsed);
        
        if (eventType.equals("FRAME_REJECTED")) {
            fields[27] = notes;
        } else if (eventType.equals("JSON_EXPORTED")) {
            fields[28] = notes;
        } else {
            fields[31] = notes;
        }

        fields[29] = DeviceInfoProvider.getDeviceModel();
        fields[30] = DeviceInfoProvider.getAndroidVersion();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 32; i++) {
            sb.append(fields[i]);
            if (i < 31) sb.append(",");
        }
        return sb.toString();
    }

    public void saveSession(Context context) {
        logEvent("SESSION_CLOSED", "");
        
        io.execute(() -> {
            String data;
            synchronized (logBuffer) {
                data = logBuffer.toString();
                logBuffer.setLength(0);
                logBuffer.append(HEADER);
            }

            try {
                String fileName = "cal_session_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".csv";
                ContentValues values = new ContentValues();
                values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
                values.put(MediaStore.MediaColumns.MIME_TYPE, "text/csv");
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/VisionProject");
                }

                Uri uri = context.getContentResolver().insert(MediaStore.Files.getContentUri("external"), values);
                if (uri != null) {
                    try (OutputStream os = context.getContentResolver().openOutputStream(uri)) {
                        if (os != null) {
                            try (PrintWriter writer = new PrintWriter(os)) {
                                writer.print(data);
                                writer.flush();
                            }
                        }
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Error saving CSV", e);
            }
        });
    }
}
