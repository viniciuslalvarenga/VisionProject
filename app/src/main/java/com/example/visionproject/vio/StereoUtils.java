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
        if (planes == null || planes.length < 3) return null;

        ByteBuffer yBuf = planes[0].getBuffer();
        ByteBuffer uBuf = planes[1].getBuffer();
        ByteBuffer vBuf = planes[2].getBuffer();

        // Always start from position 0
        yBuf.rewind(); uBuf.rewind(); vBuf.rewind();

        int yRowStride    = planes[0].getRowStride();
        int uvRowStride   = planes[1].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();

        byte[] nv21 = new byte[w * h * 3 / 2];

        // Copy Y plane row by row to strip padding
        int yLen = yBuf.remaining();
        byte[] yData = new byte[yLen];
        yBuf.get(yData);
        for (int row = 0; row < h; row++) {
            int srcPos = row * yRowStride;
            if (srcPos + w > yLen) break;
            System.arraycopy(yData, srcPos, nv21, row * w, w);
        }

        // Build interleaved VU (NV21) from UV planes
        int uLen = uBuf.remaining();
        int vLen = vBuf.remaining();
        byte[] uData = new byte[uLen];
        byte[] vData = new byte[vLen];
        uBuf.get(uData);
        vBuf.get(vData);

        int offset = w * h;
        for (int row = 0; row < h / 2; row++) {
            for (int col = 0; col < w / 2; col++) {
                int idx = row * uvRowStride + col * uvPixelStride;
                if (idx >= vLen || idx >= uLen) break;
                nv21[offset++] = vData[idx];
                nv21[offset++] = uData[idx];
            }
        }

        Mat yuv = new Mat(h * 3 / 2, w, CvType.CV_8UC1);
        yuv.put(0, 0, nv21);
        Mat rgb = new Mat();
        Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21);
        yuv.release();
        return rgb;
    }
}
