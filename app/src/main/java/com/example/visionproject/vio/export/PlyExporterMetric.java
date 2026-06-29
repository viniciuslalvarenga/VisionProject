package com.example.visionproject.vio.export;

import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import org.opencv.core.Mat;

import java.io.OutputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class PlyExporterMetric {
    private static final String TAG = "PlyExporterMetric";

    public static String export(Context context, Mat xyz, String suffix) {
        int rows = xyz.rows(), cols = xyz.cols();

        // Count valid points
        int validPoints = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double[] v = xyz.get(r, c);
                if (v != null && v.length >= 3) {
                    double z = v[2];
                    if (!Double.isInfinite(z) && !Double.isNaN(z) && z > 0 && z < 1000) {
                        validPoints++;
                    }
                }
            }
        }

        String ts = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String fileName = "vio_cloud_" + suffix + "_" + ts + ".ply";

        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        values.put(MediaStore.MediaColumns.MIME_TYPE, "application/octet-stream");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.MediaColumns.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/VisionProject");
        }

        try {
            Uri uri = context.getContentResolver().insert(
                    MediaStore.Files.getContentUri("external"), values);
            if (uri == null) { Log.e(TAG, "ContentResolver null"); return null; }

            OutputStream os = context.getContentResolver().openOutputStream(uri);
            if (os == null) { Log.e(TAG, "OutputStream null"); return null; }

            try (PrintWriter w = new PrintWriter(os)) {
                w.println("ply");
                w.println("format ascii 1.0");
                w.println("element vertex " + validPoints);
                w.println("property float x");
                w.println("property float y");
                w.println("property float z");
                w.println("end_header");

                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        double[] v = xyz.get(r, c);
                        if (v != null && v.length >= 3) {
                            double z = v[2];
                            if (!Double.isInfinite(z) && !Double.isNaN(z) && z > 0 && z < 1000) {
                                w.printf(Locale.US, "%.4f %.4f %.4f\n", v[0], v[1], v[2]);
                            }
                        }
                    }
                }
                w.flush();
            }
            return fileName;
        } catch (Exception e) {
            Log.e(TAG, "Erro ao exportar PLY", e);
            return null;
        }
    }
}
