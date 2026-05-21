package com.example.visionproject.calibracao.repository;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import com.example.visionproject.calibracao.model.CalibrationResult;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Persistência do resultado da calibração em formato JSON.
 */
public class CalibrationJsonStore {
    private static final String TAG = "CalibrationJsonStore";
    private static final String FILE_NAME = "calibration.json";

    public static String save(CalibrationResult result, Context context) {
        try {
            JSONObject json = result.toJson();
            String jsonString = json.toString(4);

            // 1. Salva no armazenamento interno (cache para os módulos usarem)
            saveToInternal(context, jsonString);

            // 2. Salva no armazenamento externo (padrão)
            File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "VisionProject");
            return saveToDir(result, context, dir);

        } catch (Exception e) {
            Log.e(TAG, "Erro ao salvar JSON de calibração", e);
            return "Error: " + e.getMessage();
        }
    }

    public static String saveToDir(CalibrationResult result, Context context, File dir) throws Exception {
        JSONObject json = result.toJson();
        String jsonString = json.toString(4);

        // Sempre salva no interno para uso da app
        saveToInternal(context, jsonString);

        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Não foi possível criar diretório: " + dir.getAbsolutePath());
        }
        
        File file = new File(dir, FILE_NAME);
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(jsonString.getBytes(StandardCharsets.UTF_8));
        }
        return file.getAbsolutePath();
    }

    private static void saveToInternal(Context context, String data) throws IOException {
        File file = new File(context.getFilesDir(), FILE_NAME);
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(data.getBytes(StandardCharsets.UTF_8));
        }
    }

    public static CalibrationResult load(Context context) {
        File file = new File(context.getFilesDir(), FILE_NAME);
        if (!file.exists()) return null;

        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            fis.read(bytes);
            String jsonString = new String(bytes, StandardCharsets.UTF_8);
            JSONObject json = new JSONObject(jsonString);

            double rms = json.getDouble("rms_reprojection_error_px");
            int used = json.getInt("images_used");
            int rejected = json.getInt("images_rejected");
            long elapsed = json.getLong("elapsed_ms_total");
            
            JSONObject imgSize = json.getJSONObject("image_size");
            Size size = new Size(imgSize.getDouble("width"), imgSize.getDouble("height"));

            JSONObject kJson = json.getJSONObject("intrinsics");
            Mat K = Mat.eye(3, 3, CvType.CV_64F);
            K.put(0, 0, kJson.getDouble("fx"));
            K.put(1, 1, kJson.getDouble("fy"));
            K.put(0, 2, kJson.getDouble("cx"));
            K.put(1, 2, kJson.getDouble("cy"));
            K.put(0, 1, kJson.getDouble("skew"));

            JSONObject dJson = json.getJSONObject("distortion");
            Mat D = new Mat(1, 5, CvType.CV_64F);
            D.put(0, 0, dJson.getDouble("k1"));
            D.put(0, 1, dJson.getDouble("k2"));
            D.put(0, 2, dJson.getDouble("p1"));
            D.put(0, 3, dJson.getDouble("p2"));
            D.put(0, 4, dJson.getDouble("k3"));

            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);
            Date date = sdf.parse(json.getString("calibration_date"));

            return new CalibrationResult(rms, K, D, new ArrayList<>(), used, rejected, elapsed, date, size);

        } catch (Exception e) {
            Log.e(TAG, "Erro ao carregar JSON de calibração", e);
            return null;
        }
    }
}
