package com.example.visionproject.vio;

import androidx.camera.core.ImageProxy;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

public class StereoUtils {

    public static Mat yuvToRgbMat(ImageProxy image) {
        int w = image.getWidth();
        int h = image.getHeight();

        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuf = planes[0].getBuffer();
        ByteBuffer uBuf = planes[1].getBuffer();
        ByteBuffer vBuf = planes[2].getBuffer();

        int yRowStride  = planes[0].getRowStride();
        int uvRowStride = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();

        byte[] yData = toByteArray(yBuf);
        byte[] uData = toByteArray(uBuf);
        byte[] vData = toByteArray(vBuf);

        // Build NV21 (Y plane then interleaved VU)
        byte[] nv21 = new byte[w * h * 3 / 2];
        for (int row = 0; row < h; row++) {
            System.arraycopy(yData, row * yRowStride, nv21, row * w, w);
        }
        int offset = w * h;
        for (int row = 0; row < h / 2; row++) {
            for (int col = 0; col < w / 2; col++) {
                nv21[offset++] = vData[row * uvRowStride + col * uvPixelStride];
                nv21[offset++] = uData[row * uvRowStride + col * uvPixelStride];
            }
        }

        Mat yuv = new Mat(h * 3 / 2, w, CvType.CV_8UC1);
        yuv.put(0, 0, nv21);
        Mat rgb = new Mat();
        Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21);
        yuv.release();
        return rgb;
    }

    private static byte[] toByteArray(ByteBuffer buf) {
        byte[] bytes = new byte[buf.remaining()];
        buf.get(bytes);
        return bytes;
    }
}
