package com.example.visionproject.vio.model;

public class Pose {
    public final long tNs;
    public final double[] p;
    public final float[] q;

    public Pose(long tNs, double[] p, float[] q) {
        this.tNs = tNs;
        this.p = p == null ? new double[]{0.0, 0.0, 0.0} : p.clone();
        this.q = q == null ? new float[]{0f, 0f, 0f, 1f} : q.clone();
    }
}
