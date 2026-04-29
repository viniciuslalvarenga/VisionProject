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
    
    private Mat mReferenceRoi;
    private volatile double mCurrentPcc = 1.0;
    private double mSmoothedPcc = 1.0; // Filtro passa-baixa para estabilidade
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
        
        // DOWNSCALING AGRESSIVO: 80x60 para ignorar ruídos finos e tremores de mão
        Mat smallGray = new Mat();
        org.opencv.core.Size smallSize = new org.opencv.core.Size(80, 60);
        Imgproc.resize(gray, smallGray, smallSize);
        
        Mat currentRoi = smallGray; // Já está na escala correta

        boolean shouldProcess = true;

        if (!mReferenceRoi.empty() && mReferenceRoi.size().equals(currentRoi.size())) {
            double rawPcc = computePCC(currentRoi, mReferenceRoi);
            
            // FILTRO TEMPORAL: 70% do valor anterior + 30% do novo
            // Isso suaviza picos causados por movimentos bruscos da mão
            mSmoothedPcc = (mSmoothedPcc * 0.7) + (rawPcc * 0.3);
            mCurrentPcc = mSmoothedPcc;
            
            mCurrentCre = (1.0 - mCurrentPcc) / (1.0 - mThresholdTheta);

            // Só sai de DISCARD se o movimento for sustentado
            if (mCurrentPcc >= mThresholdTheta) {
                mDiscardedFrames.incrementAndGet();
                mStatus = (mCurrentPcc >= THETA_IDLE) ? "IDLE" : "DISCARD";
                shouldProcess = false;
                
                // Atualização adaptativa da referência
                if (mCurrentPcc >= THETA_IDLE) {
                    mReferenceRoi.release();
                    mReferenceRoi = currentRoi.clone();
                }
            } else {
                mStatus = "PROCESS";
                mReferenceRoi.release();
                mReferenceRoi = currentRoi.clone();
            }
        } else {
            mReferenceRoi = currentRoi.clone();
            mStatus = "START";
            mSmoothedPcc = 1.0;
        }

        // Feedback Visual de borda grossa (retângulo externo)
        Scalar boxColor = mStatus.equals("PROCESS") ? new Scalar(0, 255, 0, 255) : 
                         mStatus.equals("DISCARD") ? new Scalar(255, 255, 0, 255) : new Scalar(255, 255, 255, 255);
        Imgproc.rectangle(rgbaFrame, new org.opencv.core.Point(0,0), 
                         new org.opencv.core.Point(rgbaFrame.cols(), rgbaFrame.rows()), boxColor, 15);

        mItaPointsAfter = shouldProcess ? (gray.rows() * gray.cols()) : 0;
        
        smallGray.release();
        gray.release();
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
