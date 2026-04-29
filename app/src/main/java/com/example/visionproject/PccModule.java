package com.example.visionproject;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

public class PccModule {
    private volatile double mThresholdTheta = 0.85; 
    private static final double THETA_IDLE = 0.97;
    
    private Mat mReferenceRoi;
    private final Mat mF1 = new Mat();
    private final Mat mF2 = new Mat();
    private final Mat mGray = new Mat();
    private final Mat mSmallGray = new Mat();
    private final Size mSmallSize = new Size(80, 60);
    
    private volatile double mCurrentPcc = 1.0;
    private double mSmoothedPcc = 1.0; 
    private volatile double mCurrentCre = 0.0;
    private final AtomicLong mTotalFrames = new AtomicLong(0);
    private final AtomicLong mDiscardedFrames = new AtomicLong(0);
    private volatile String mStatus = "IDLE";
    private volatile int mItaPointsAfter = 0;

    private final Scalar mColorProcess = new Scalar(0, 255, 0, 255);
    private final Scalar mColorDiscard = new Scalar(255, 255, 0, 255);
    private final Scalar mColorIdle = new Scalar(255, 255, 255, 255);

    public PccModule() {
        mReferenceRoi = new Mat();
    }

    private double computePCC(Mat m1, Mat m2) {
        if (m1.empty() || m2.empty()) return 1.0;
        
        m1.convertTo(mF1, CvType.CV_32F);
        m2.convertTo(mF2, CvType.CV_32F);
        
        Scalar mean1 = Core.mean(mF1);
        Scalar mean2 = Core.mean(mF2);
        Core.subtract(mF1, mean1, mF1);
        Core.subtract(mF2, mean2, mF2);
        
        double numerator = mF1.dot(mF2);
        double den1 = mF1.dot(mF1);
        double den2 = mF2.dot(mF2);
        
        double pcc = 1.0;
        double denominator = Math.sqrt(den1 * den2);
        if (denominator > 1e-10) {
            pcc = numerator / denominator;
        }
        
        return pcc;
    }

    public boolean processFrame(Mat rgbaFrame) {
        mTotalFrames.incrementAndGet();
        
        Imgproc.cvtColor(rgbaFrame, mGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.resize(mGray, mSmallGray, mSmallSize);
        
        boolean shouldProcess = true;

        if (!mReferenceRoi.empty() && mReferenceRoi.size().equals(mSmallGray.size())) {
            double rawPcc = computePCC(mSmallGray, mReferenceRoi);
            
            mSmoothedPcc = (mSmoothedPcc * 0.7) + (rawPcc * 0.3);
            mCurrentPcc = mSmoothedPcc;
            
            mCurrentCre = (1.0 - mCurrentPcc) / (1.0 - mThresholdTheta);

            if (mCurrentPcc >= mThresholdTheta) {
                mDiscardedFrames.incrementAndGet();
                mStatus = (mCurrentPcc >= THETA_IDLE) ? "IDLE" : "DISCARD";
                shouldProcess = false;
                
                if (mCurrentPcc >= THETA_IDLE) {
                    mSmallGray.copyTo(mReferenceRoi);
                }
            } else {
                mStatus = "PROCESS";
                mSmallGray.copyTo(mReferenceRoi);
            }
        } else {
            mSmallGray.copyTo(mReferenceRoi);
            mStatus = "START";
            mSmoothedPcc = 1.0;
        }

        Scalar boxColor = mStatus.equals("PROCESS") ? mColorProcess : 
                         mStatus.equals("DISCARD") ? mColorDiscard : mColorIdle;
        
        Imgproc.rectangle(rgbaFrame, new org.opencv.core.Point(0,0), 
                         new org.opencv.core.Point(rgbaFrame.cols(), rgbaFrame.rows()), boxColor, 15);

        mItaPointsAfter = shouldProcess ? (mGray.rows() * mGray.cols()) : 0;
        
        return shouldProcess;
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
