package com.example.visionproject.vio.repository;

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
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class VioCsvLogger {
    private static final String TAG = "VioCsvLogger";
    private static VioCsvLogger instance;

    private final List<String> logs = Collections.synchronizedList(new ArrayList<>());
    private final String sessionId;
    private final SimpleDateFormat isoFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US);

    private VioCsvLogger() {
        this.sessionId = "VIO_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        logs.add("session_id,timestamp_ms,timestamp_iso,event_type," +
                "scene_id,frame_tns,sensor_type," +
                "ax,ay,az,gx,gy,gz,qx,qy,qz,qw," +
                "baseline_m,parallax_px,matches_good,inliers_ransac,rotation_deg," +
                "fx_px,Z_est_m,Z_real_m,error_pct," +
                "file_path,device_model,android_version,notes");
        log("SESSION_START", "Device: " + Build.MANUFACTURER + " " + Build.MODEL);
    }

    public static synchronized VioCsvLogger getInstance() {
        if (instance == null) instance = new VioCsvLogger();
        return instance;
    }

    private static String esc(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n"))
            return "\"" + s.replace("\"", "\"\"") + "\"";
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
        String line = String.format(Locale.US,
                "%s,%d,%s,%s,,,,,,,,,,,,,,,,,,,,,,,,%s,%d,%s",
                sessionId, ms, iso, esc(eventType),
                esc(device), Build.VERSION.SDK_INT, esc(notes));
        logs.add(line);
        Log.d(TAG, eventType + (notes != null ? ": " + notes : ""));
    }

    public void logDetailed(String eventType, String sceneId, Long frameTns, String sensorType,
                            Double ax, Double ay, Double az,
                            Double gx, Double gy, Double gz,
                            Double qx, Double qy, Double qz, Double qw,
                            Double baselineM, Double parallaxPx, Integer matchesGood,
                            Integer inliersRansac, Double rotationDeg,
                            Double fxPx, Double zEstM, Double zRealM, Double errorPct,
                            String filePath, String notes) {
        String iso = isoFormat.format(new Date());
        long ms = System.currentTimeMillis();
        String device = Build.MANUFACTURER + " " + Build.MODEL;

        // Use a StringBuilder or explicit concatenation to avoid format string mismatches with %d vs Long/String
        StringBuilder sb = new StringBuilder();
        sb.append(sessionId).append(",");
        sb.append(ms).append(",");
        sb.append(iso).append(",");
        sb.append(esc(eventType)).append(",");
        sb.append(esc(sceneId)).append(",");
        sb.append(frameTns != null ? frameTns.toString() : "").append(",");
        sb.append(esc(sensorType)).append(",");
        sb.append(f(ax,4)).append(",");
        sb.append(f(ay,4)).append(",");
        sb.append(f(az,4)).append(",");
        sb.append(f(gx,4)).append(",");
        sb.append(f(gy,4)).append(",");
        sb.append(f(gz,4)).append(",");
        sb.append(f(qx,4)).append(",");
        sb.append(f(qy,4)).append(",");
        sb.append(f(qz,4)).append(",");
        sb.append(f(qw,4)).append(",");
        sb.append(f(baselineM,4)).append(",");
        sb.append(f(parallaxPx,1)).append(",");
        sb.append(n(matchesGood)).append(",");
        sb.append(n(inliersRansac)).append(",");
        sb.append(f(rotationDeg,2)).append(",");
        sb.append(f(fxPx,2)).append(",");
        sb.append(f(zEstM,4)).append(",");
        sb.append(f(zRealM,4)).append(",");
        sb.append(f(errorPct,2)).append(",");
        sb.append(esc(filePath)).append(",");
        sb.append(esc(device)).append(",");
        sb.append(Build.VERSION.SDK_INT).append(",");
        sb.append(esc(notes));

        logs.add(sb.toString());
        Log.d(TAG, eventType);
    }

    public void saveSession(Context context) {
        new Thread(() -> {
            String fileName = "vio_session_" + sessionId + ".csv";
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
                if (uri == null) { Log.e(TAG, "ContentResolver returned null"); return; }
                OutputStream os = context.getContentResolver().openOutputStream(uri);
                if (os == null) { Log.e(TAG, "OutputStream is null"); return; }
                List<String> logsCopy;
                synchronized (logs) {
                    logsCopy = new ArrayList<>(logs);
                }
                try (PrintWriter writer = new PrintWriter(os)) {
                    for (String line : logsCopy) writer.println(line);
                    writer.flush();
                }
                showToast(context, "VIO CSV salvo: " + fileName);
            } catch (Exception e) {
                Log.e(TAG, "Erro ao salvar CSV", e);
                showToast(context, "Erro ao salvar CSV: " + e.getMessage());
            }
        }).start();
    }

    private void showToast(Context ctx, String msg) {
        android.os.Handler h = new android.os.Handler(android.os.Looper.getMainLooper());
        h.post(() -> android.widget.Toast.makeText(ctx, msg, android.widget.Toast.LENGTH_LONG).show());
    }

    public String getSessionId() { return sessionId; }
}
