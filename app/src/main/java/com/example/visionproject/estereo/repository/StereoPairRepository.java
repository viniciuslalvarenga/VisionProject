package com.example.visionproject.estereo.repository;

import com.example.visionproject.estereo.model.StereoPair;
import org.opencv.core.Mat;

public class StereoPairRepository {
    private static StereoPairRepository instance;
    private StereoPair currentPair;
    private Mat lastDisparity;

    private StereoPairRepository() {}

    public static synchronized StereoPairRepository getInstance() {
        if (instance == null) instance = new StereoPairRepository();
        return instance;
    }

    public synchronized void setCurrentPair(StereoPair pair) {
        this.currentPair = pair;
    }

    public synchronized StereoPair getCurrentPair() {
        return currentPair;
    }

    public synchronized void setLastDisparity(Mat disparity) {
        if (this.lastDisparity != null) this.lastDisparity.release();
        this.lastDisparity = disparity != null ? disparity.clone() : null;
    }

    public synchronized Mat getLastDisparity() {
        return lastDisparity;
    }

    public synchronized void clear() {
        currentPair = null;
        if (lastDisparity != null) {
            lastDisparity.release();
            lastDisparity = null;
        }
    }
}
