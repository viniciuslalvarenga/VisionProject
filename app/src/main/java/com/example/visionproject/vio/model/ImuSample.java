package com.example.visionproject.vio.model;

public class ImuSample {
    public final long tNs;
    public final float[] values;
    public final int sensorType;

    public ImuSample(long tNs, float[] values, int sensorType) {
        this.tNs = tNs;
        this.values = values == null ? new float[0] : values.clone();
        this.sensorType = sensorType;
    }
}
