package com.example.visionproject;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

public class PccModule {
    private volatile double mThresholdTheta = 0.85; 
    public static final double THETA_IDLE = 0.97;
    private static final double Rc = 0.40;
    
    private Mat mReferenceRoi;
    private volatile double mCurrentPcc = 1.0;
    private volatile double mCurrentCre = 0.0;
    private final AtomicLong mTotalFrames = new AtomicLong(0);
    private final AtomicLong mDiscardedFrames = new AtomicLong(0);
    private volatile String mStatus = "IDLE";
    private volatile int mItaPointsAfter = 0;

    public PccModule() {
        mReferenceRoi = new Mat();
    }

    public double computePCC(Mat m1, Mat m2) {
        if (m1.empty() || m2.empty()) return 1.0;
        
        Mat f1 = new Mat(), f2 = new Mat();
        m1.convertTo(f1, CvType.CV_32F);
        m2.convertTo(f2, CvType.CV_32F);
        
        Scalar mean1 = Core.mean(f1);
        Scalar mean2 = Core.mean(f2);
        Core.subtract(f1, mean1, f1);
        Core.subtract(f2, mean2, f2);
        
        double numerator = f1.dot(f2);
        double den1 = f1.dot(f1);
        double den2 = f2.dot(f2);
        
        double pcc = 1.0;
        double denominator = Math.sqrt(den1 * den2);
        if (denominator > 1e-10) {
            pcc = numerator / denominator;
        }
        
        f1.release();
        f2.release();
        return pcc;
    }

    public boolean processFrame(Mat rgbaFrame) {
        mTotalFrames.incrementAndGet();
        
        Mat gray = new Mat();
        Imgproc.cvtColor(rgbaFrame, gray, Imgproc.COLOR_RGBA2GRAY);
        
        int width = gray.cols();
        int height = gray.rows();
        Rect roiRect = new Rect(0, 0, width, height);
        Mat currentRoi = new Mat(gray, roiRect);

        boolean shouldProcess = true;

        if (!mReferenceRoi.empty() && mReferenceRoi.size().equals(currentRoi.size())) {
            mCurrentPcc = computePCC(currentRoi, mReferenceRoi);
            mCurrentCre = Rc / (1.0001 - Math.abs(mCurrentPcc));

            applyRoiOverlay(rgbaFrame, currentRoi, mReferenceRoi, roiRect);

            if (mCurrentPcc >= mThresholdTheta) {
                mDiscardedFrames.incrementAndGet();
                mStatus = (mCurrentPcc >= THETA_IDLE) ? "IDLE" : "DISCARD";
                shouldProcess = false;
            } else {
                mStatus = "PROCESS";
                mReferenceRoi.release();
                mReferenceRoi = currentRoi.clone();
            }
        } else {
            mReferenceRoi = currentRoi.clone();
            mStatus = "START";
        }

        Scalar boxColor = mStatus.equals("PROCESS") ? new Scalar(0, 255, 0, 255) : 
                         mStatus.equals("DISCARD") ? new Scalar(255, 255, 0, 255) : new Scalar(255, 255, 255, 255);
        Imgproc.rectangle(rgbaFrame, roiRect.tl(), roiRect.br(), boxColor, 4);

        mItaPointsAfter = (int) (currentRoi.rows() * currentRoi.cols() * (1.0 - Math.abs(mCurrentPcc)));
        
        currentRoi.release();
        gray.release();
        return shouldProcess;
    }

    private void applyRoiOverlay(Mat rgbaFrame, Mat curr, Mat ref, Rect roiRect) {
        Mat diff = new Mat();
        Core.absdiff(curr, ref, diff);
        Imgproc.threshold(diff, diff, 25, 255, Imgproc.THRESH_BINARY);
        Mat roiSubmat = rgbaFrame.submat(roiRect);
        Mat redOverlay = new Mat(roiSubmat.size(), roiSubmat.type(), new Scalar(255, 0, 0, 150));
        redOverlay.copyTo(roiSubmat, diff);
        roiSubmat.release();
        redOverlay.release();
        diff.release();
    }

    public double getDiscardRate() {
        long total = mTotalFrames.get();
        if (total == 0) return 0;
        return (double) mDiscardedFrames.get() / total * 100.0;
    }

    public String getFullStatus() {
        return String.format(Locale.US, "%s (%d/%d)", mStatus, mDiscardedFrames.get(), mTotalFrames.get());
    }

    public void resetStats() { 
        mTotalFrames.set(0); 
        mDiscardedFrames.set(0); 
        mStatus = "IDLE";
        if (mReferenceRoi != null) mReferenceRoi.release();
        mReferenceRoi = new Mat();
    }

    public void setThresholdTheta(double t) { this.mThresholdTheta = t; }
    public double getThresholdTheta() { return mThresholdTheta; }
    public String getStatus() { return mStatus; }
    public double getCurrentPcc() { return mCurrentPcc; }
    public double getCurrentCre() { return mCurrentCre; }
    public int getItaPointsAfter() { return mItaPointsAfter; }
}
