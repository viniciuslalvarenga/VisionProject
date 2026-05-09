package com.example.visionproject.shared;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
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
 * Helper para exportar imagens para o MediaStore.
 */
public final class FileExporter {

    private static final String TAG = "FileExporter";

    private FileExporter() {}

    /**
     * Salva um Mat do OpenCV como PNG na pasta Pictures/VisionProject.
     */
    public static Uri saveMatAsPng(Context ctx, Mat mat, String filename) {
        if (mat == null || mat.empty()) return null;
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        return saveBitmapAsPng(ctx, bitmap, filename);
    }

    /**
     * Salva um Bitmap como PNG na pasta Pictures/VisionProject.
     */
    public static Uri saveBitmapAsPng(Context ctx, Bitmap bitmap, String filename) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename + ".png");
        values.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/VisionProject");
        }

        Uri uri = ctx.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri != null) {
            try (OutputStream os = ctx.getContentResolver().openOutputStream(uri)) {
                if (os != null) {
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, os);
                    return uri;
                }
            } catch (Exception e) {
                Log.e(TAG, "Erro ao salvar bitmap", e);
            }
        }
        return null;
    }

    /**
     * Abre o seletor de compartilhamento para um Uri de imagem.
     */
    public static void shareImage(Context ctx, Uri imageUri) {
        if (imageUri == null) return;
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType("image/png");
        intent.putExtra(Intent.EXTRA_STREAM, imageUri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        ctx.startActivity(Intent.createChooser(intent, "Compartilhar Imagem"));
    }
}
