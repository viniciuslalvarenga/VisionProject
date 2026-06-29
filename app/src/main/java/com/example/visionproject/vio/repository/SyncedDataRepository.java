package com.example.visionproject.vio.repository;

import com.example.visionproject.vio.model.FrameSample;
import com.example.visionproject.vio.model.ImuSample;

import java.util.ArrayDeque;
import java.util.Deque;

public class SyncedDataRepository {
    private static final int MAX_FRAMES = 30;
    private static final int MAX_IMU    = 5000;

    private static SyncedDataRepository instance;

    private final Deque<FrameSample> frames = new ArrayDeque<>();
    private final Deque<ImuSample>   imuSamples = new ArrayDeque<>();

    private SyncedDataRepository() {}

    public static synchronized SyncedDataRepository getInstance() {
        if (instance == null) instance = new SyncedDataRepository();
        return instance;
    }

    public synchronized void addFrame(FrameSample f) {
        if (frames.size() >= MAX_FRAMES) {
            FrameSample old = frames.pollFirst();
            if (old != null) old.release();
        }
        frames.addLast(f);
    }

    public synchronized void addImuSample(ImuSample s) {
        if (imuSamples.size() >= MAX_IMU) imuSamples.pollFirst();
        imuSamples.addLast(s);
    }

    public synchronized FrameSample getLatestFrame() {
        return frames.isEmpty() ? null : frames.peekLast();
    }

    public synchronized void clear() {
        for (FrameSample f : frames) f.release();
        frames.clear();
        imuSamples.clear();
    }
}
