package com.example.visionproject.estereo.export;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.OutputStream;

/**
 * Helper para exportar artefatos PNG do pipeline estereo para Pictures/VisionProject/estereo/.
 * Reutiliza padrao MediaStore consistente com modulos PCC, ModeloCamera e Calibracao.
 */
public class ImageExporter {
    private static final String TAG = "ImageExporter";
    private static final String SUBDIR = "VisionProject/estereo";

    /**
     * Salva uma Mat (BGR ou GRAY) como PNG no MediaStore.
     * Retorna a Uri salva ou null em caso de erro.
     */
    public static Uri saveMat(Context ctx, Mat mat, String filename) {
        if (mat == null || mat.empty()) return null;
        try {
            Bitmap bmp = matToBitmap(mat);
            return saveBitmap(ctx, bmp, filename);
        } catch (Exception e) {
            Log.e(TAG, "Erro ao salvar " + filename, e);
            return null;
        }
    }

    public static Uri saveBitmap(Context ctx, Bitmap bmp, String filename) {
        if (bmp == null) return null;
        ContentValues v = new ContentValues();
        v.put(MediaStore.Images.Media.DISPLAY_NAME, filename);
        v.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            v.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + SUBDIR);
        }
        Uri uri = ctx.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v);
        if (uri == null) return null;
        try (OutputStream out = ctx.getContentResolver().openOutputStream(uri)) {
            if (out != null) {
                bmp.compress(Bitmap.CompressFormat.PNG, 100, out);
            }
            return uri;
        } catch (Exception e) {
            Log.e(TAG, "Erro ao escrever PNG " + filename, e);
            return null;
        }
    }

    private static Bitmap matToBitmap(Mat mat) {
        Mat display = mat;
        // Converter para 3 canais se for grayscale
        if (mat.channels() == 1) {
            display = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(mat, display, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR);
        }
        Bitmap bmp = Bitmap.createBitmap(display.cols(), display.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(display, bmp);
        if (display != mat) display.release();
        return bmp;
    }
}
