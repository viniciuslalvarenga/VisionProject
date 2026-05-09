package com.example.visionproject.modelocamera.logger;

import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import com.example.visionproject.modelocamera.logger.event.SessionClosedEvent;
import com.example.visionproject.shared.DeviceInfoProvider;
import com.example.visionproject.shared.csv.CsvFormatter;
import com.example.visionproject.shared.csv.CsvWriter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Singleton responsável por gerenciar a sessão de log em CSV para o Módulo 2.
 */
public class CalibrationSessionLogger {

    private static final String TAG = "CalibrationLogger";

    private static class Holder {
        static final CalibrationSessionLogger INSTANCE = new CalibrationSessionLogger();
    }

    public static CalibrationSessionLogger getInstance() {
        return Holder.INSTANCE;
    }

    private static final String[] CSV_COLUMNS = {
            "session_id", "timestamp_ms", "timestamp_iso", "event_type",
            "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
            "res_w", "res_h", "fov_h_deg", "distortion_type",
            "x_orig", "y_orig", "x_corr", "y_corr", "delta_x", "delta_y", "dist_from_center",
            "color_hex", "point_index",
            "image_filename", "image_source",
            "elapsed_ms",
            "device_model", "android_version", "notes"
    };

    private CsvWriter writer;
    private String sessionId;
    private ExecutorService io = Executors.newSingleThreadExecutor();
    private final AtomicInteger eventCount = new AtomicInteger(0);
    private boolean active = false;
    private final CsvFormatter formatter = new CsvFormatter();

    private CalibrationSessionLogger() {
    }

    public synchronized void startSession(Context ctx) {
        if (active) {
            endSession(ctx, "RESTARTED");
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        sessionId = "mc_" + timeStamp;
        String fileName = "mc_session_" + timeStamp + ".csv";

        try {
            OutputStream os = getOutputStream(ctx, fileName);
            if (os != null) {
                writer = new CsvWriter(os, CSV_COLUMNS);
                writer.writeHeader();
                active = true;
                eventCount.set(0);
                logInternal("SESSION_START", "App started");
            }
        } catch (IOException e) {
            Log.e(TAG, "Erro ao iniciar sessão de log", e);
        }
    }

    private OutputStream getOutputStream(Context ctx, String fileName) throws IOException {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
            values.put(MediaStore.MediaColumns.MIME_TYPE, "text/csv");
            values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS + "/VisionProject");

            Uri uri = ctx.getContentResolver().insert(MediaStore.Files.getContentUri("external"), values);
            if (uri != null) {
                return ctx.getContentResolver().openOutputStream(uri);
            }
        } else {
            File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "VisionProject");
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IOException("Não foi possível criar o diretório: " + dir.getAbsolutePath());
            }
            return new FileOutputStream(new File(dir, fileName));
        }
        return null;
    }

    public synchronized void log(SessionEvent event) {
        if (!active) {
            Log.w(TAG, "Tentativa de log em sessão inativa");
            return;
        }

        Map<String, String> baseFields = createBaseFields(event.timestampMs, event.eventType);
        baseFields.putAll(event.toCsvFields(formatter));
        enqueueWrite(baseFields);
        eventCount.incrementAndGet();
    }

    private void logInternal(String type, String notes) {
        long now = System.currentTimeMillis();
        Map<String, String> fields = createBaseFields(now, type);
        fields.put("notes", notes);
        enqueueWrite(fields);
        eventCount.incrementAndGet();
    }

    private Map<String, String> createBaseFields(long timestampMs, String eventType) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("session_id", sessionId);
        m.put("timestamp_ms", String.valueOf(timestampMs));
        m.put("timestamp_iso", CsvFormatter.fmtTimestampIso(timestampMs));
        m.put("event_type", eventType);
        m.put("device_model", DeviceInfoProvider.getDeviceModel());
        m.put("android_version", DeviceInfoProvider.getAndroidVersion());
        return m;
    }

    private void enqueueWrite(Map<String, String> fields) {
        io.execute(() -> {
            try {
                if (writer != null) {
                    writer.writeRow(fields);
                }
            } catch (IOException e) {
                Log.e(TAG, "Erro ao escrever linha no CSV", e);
            }
        });
    }

    public synchronized void endSession(Context ctx, String reason) {
        if (!active) return;

        log(new SessionClosedEvent(eventCount.get(), reason));
        
        active = false;
        io.execute(() -> {
            try {
                if (writer != null) {
                    writer.close();
                    writer = null;
                }
            } catch (IOException e) {
                Log.e(TAG, "Erro ao fechar o CSV", e);
            }
        });

        io.shutdown();
        io = Executors.newSingleThreadExecutor();
    }

    public boolean isActive() {
        return active;
    }
}
