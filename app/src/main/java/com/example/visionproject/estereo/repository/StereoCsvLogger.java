package com.example.visionproject.estereo.repository;

import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import java.io.OutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class StereoCsvLogger {
    private static final String TAG = "StereoCsvLogger";
    private static StereoCsvLogger instance;
    private final List<String> logs = new ArrayList<>();
    private final String sessionId;
    private final SimpleDateFormat isoFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US);

    private StereoCsvLogger() {
        this.sessionId = "EST_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        logs.add("session_id,timestamp_ms,timestamp_iso,event_type,scene_id,side,baseline_mm,matches_raw,matches_good,inliers_ransac,sgbm_preset,num_disparities,block_size,elapsed_ms,pct_valid,fx_px,reference_distance_m,measured_z_m,error_pct,file_path,device_model,android_version,notes");
        log("SESSION_START", "App started: " + Build.MANUFACTURER + " " + Build.MODEL);
    }

    public static synchronized StereoCsvLogger getInstance() {
        if (instance == null) instance = new StereoCsvLogger();
        return instance;
    }

    private static String esc(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }

    private static String n(Number x) { return x == null ? "" : x.toString(); }
    private static String f(Double x, int dec) {
        if (x == null || Double.isNaN(x) || Double.isInfinite(x)) return "";
        return String.format(Locale.US, "%." + dec + "f", x);
    }

    public void log(String eventType, String notes) {
        String iso = isoFormat.format(new Date());
        long ms = System.currentTimeMillis();
        String device = Build.MANUFACTURER + " " + Build.MODEL;
        String line = String.format(Locale.US, "%s,%d,%s,%s,,,,,,,,,,,,,,,,,%s,%d,%s",
                sessionId, ms, iso, eventType, esc(device), Build.VERSION.SDK_INT, esc(notes));
        logs.add(line);
        Log.d(TAG, "Log: " + eventType);
    }

    public void logDetailed(String eventType, String sceneId, String side, Integer baselineMm,
                             Integer matchesRaw, Integer matchesGood, Integer inliersRansac,
                             String sgbmPreset, Integer numDisp, Integer blockSize,
                             Long elapsedMs, Double pctValid, Double fxPx,
                             Double refDistanceM, Double measuredZm, Double errorPct,
                             String filePath, String notes) {
        String iso = isoFormat.format(new Date());
        long ms = System.currentTimeMillis();
        String device = Build.MANUFACTURER + " " + Build.MODEL;
        String line = String.format(Locale.US,
                "%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s",
                sessionId, ms, iso, eventType,
                esc(sceneId), esc(side), n(baselineMm),
                n(matchesRaw), n(matchesGood), n(inliersRansac),
                esc(sgbmPreset), n(numDisp), n(blockSize),
                n(elapsedMs), f(pctValid, 4), f(fxPx, 2),
                f(refDistanceM, 4), f(measuredZm, 4), f(errorPct, 2),
                esc(filePath), esc(device), Build.VERSION.SDK_INT, esc(notes));
        logs.add(line);
        Log.d(TAG, "LogDetailed: " + eventType);
    }

    public void saveSession(Context context) {
        // Executar em thread background para não travar UI
        new Thread(() -> {
            String fileName = "est_session_" + sessionId + ".csv";
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
            values.put(MediaStore.MediaColumns.MIME_TYPE, "text/csv");
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.MediaColumns.RELATIVE_PATH,
                        Environment.DIRECTORY_DOCUMENTS + "/VisionProject");
            }
            try {
                Uri uri = context.getContentResolver().insert(
                        MediaStore.Files.getContentUri("external"), values);
                if (uri == null) {
                    String msg = "Erro: Nao foi possivel criar arquivo CSV (ContentResolver retornou null)";
                    Log.e(TAG, msg);
                    showToastOnUIThread(context, msg);
                    return;
                }
                
                OutputStream os = context.getContentResolver().openOutputStream(uri);
                if (os == null) {
                    String msg = "Erro: Nao foi possivel abrir stream para CSV (OutputStream eh null)";
                    Log.e(TAG, msg);
                    showToastOnUIThread(context, msg);
                    return;
                }
                
                try (PrintWriter writer = new PrintWriter(os)) {
                    for (String line : logs) {
                        writer.println(line);
                    }
                    writer.flush();
                    String msg = "CSV salvo: " + fileName;
                    Log.d(TAG, msg);
                    showToastOnUIThread(context, msg);
                }
            } catch (Exception e) {
                String msg = "Erro ao salvar CSV: " + e.getMessage();
                Log.e(TAG, msg, e);
                showToastOnUIThread(context, msg);
            }
        }).start();
    }

    private void showToastOnUIThread(Context context, String message) {
        if (context != null) {
            android.os.Handler handler = new android.os.Handler(android.os.Looper.getMainLooper());
            handler.post(() -> android.widget.Toast.makeText(context, message, android.widget.Toast.LENGTH_LONG).show());
        }
    }
}
