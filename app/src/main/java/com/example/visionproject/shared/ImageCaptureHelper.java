package com.example.visionproject.shared;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.MediaStore;

import androidx.activity.result.ActivityResultLauncher;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;

/**
 * Helper para captura de imagem via câmera ou galeria.
 */
public final class ImageCaptureHelper {

    private ImageCaptureHelper() {}

    /**
     * Lança o seletor da galeria.
     */
    public static void launchGalleryPicker(ActivityResultLauncher<Intent> launcher) {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        launcher.launch(intent);
    }

    /**
     * Converte um Uri de imagem para um Mat do OpenCV (formato RGBA).
     */
    public static Mat uriToMat(Context ctx, Uri uri) {
        try (InputStream is = ctx.getContentResolver().openInputStream(uri)) {
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            if (bitmap == null) return new Mat();
            
            Mat mat = new Mat();
            Utils.bitmapToMat(bitmap, mat);
            // Garantir que está em RGBA
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGBA);
            return mat;
        } catch (Exception e) {
            return new Mat();
        }
    }
}
