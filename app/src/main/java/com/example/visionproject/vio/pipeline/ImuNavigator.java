package com.example.visionproject.vio.pipeline;

import android.hardware.SensorManager;

import com.example.visionproject.vio.model.Pose;
import com.example.visionproject.vio.strategy.ZuptStrategy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ImuNavigator {
    private static ImuNavigator instance;

    private final double[] p = {0, 0, 0};
    private final double[] v = {0, 0, 0};
    private float[] q = {0, 0, 0, 1};
    private long tPrev = -1;
    private double restAccum = 0;

    private static final double EPS    = 0.08;
    private static final double DT_MAX = 0.1;

    private final List<Pose> history = new ArrayList<>();
    private final ZuptStrategy zupt;

    public ImuNavigator(ZuptStrategy zupt) {
        this.zupt = zupt;
    }

    public static synchronized ImuNavigator getInstance() {
        if (instance == null) instance = new ImuNavigator(new ZuptStrategy.DefaultZuptStrategy());
        return instance;
    }

    public synchronized void onAccel(long tNs, float[] aBody, float[] qNow) {
        q = qNow.clone();
        if (tPrev < 0) { tPrev = tNs; return; }
        double dt = (tNs - tPrev) * 1e-9;
        tPrev = tNs;
        if (dt <= 0 || dt > DT_MAX) return;

        double[] aNav = rotateBodyToNav(aBody, q);
        double mag = Math.sqrt(aBody[0] * aBody[0] + aBody[1] * aBody[1] + aBody[2] * aBody[2]);

        if (zupt.isAtRest(mag, dt, restAccum)) {
            v[0] = v[1] = v[2] = 0;
            restAccum = 0;
        } else if (mag < EPS) {
            restAccum += dt;
        } else {
            restAccum = 0;
        }

        for (int i = 0; i < 3; i++) {
            p[i] += v[i] * dt + 0.5 * aNav[i] * dt * dt;
            v[i] += aNav[i] * dt;
        }
        history.add(new Pose(tNs, p.clone(), q.clone()));
    }

    public synchronized double distanceBetween(long tA, long tB) {
        double[] pa = posAt(tA);
        double[] pb = posAt(tB);
        return Math.sqrt(sq(pb[0]-pa[0]) + sq(pb[1]-pa[1]) + sq(pb[2]-pa[2]));
    }

    public synchronized Pose getPoseAt(long tNs) {
        if (history.isEmpty()) return new Pose(tNs, new double[]{0,0,0}, new float[]{0,0,0,1});
        // Linear interpolation
        Pose prev = history.get(0), next = history.get(0);
        for (Pose pose : history) {
            if (pose.tNs <= tNs) prev = pose;
            if (pose.tNs >= tNs) { next = pose; break; }
        }
        if (prev.tNs == next.tNs) return prev;
        double t = (double)(tNs - prev.tNs) / (next.tNs - prev.tNs);
        double[] pInterp = {
            prev.p[0] + t * (next.p[0] - prev.p[0]),
            prev.p[1] + t * (next.p[1] - prev.p[1]),
            prev.p[2] + t * (next.p[2] - prev.p[2])
        };
        return new Pose(tNs, pInterp, prev.q);
    }

    public synchronized void resetWithPose(double[] pNew, float[] qNew) {
        System.arraycopy(pNew, 0, p, 0, 3);
        v[0] = v[1] = v[2] = 0;
        q = qNew.clone();
    }

    public synchronized void reset() {
        p[0] = p[1] = p[2] = 0;
        v[0] = v[1] = v[2] = 0;
        q = new float[]{0, 0, 0, 1};
        tPrev = -1;
        restAccum = 0;
        history.clear();
    }

    public synchronized List<Pose> getHistoryCopy() { return new ArrayList<>(history); }

    private double[] posAt(long tNs) {
        if (history.isEmpty()) return new double[]{0, 0, 0};
        Pose best = history.get(0);
        long minDiff = Math.abs(tNs - best.tNs);
        for (Pose pose : history) {
            long diff = Math.abs(tNs - pose.tNs);
            if (diff < minDiff) { minDiff = diff; best = pose; }
        }
        return best.p.clone();
    }

    private double[] rotateBodyToNav(float[] a, float[] qn) {
        float[] R = new float[9];
        SensorManager.getRotationMatrixFromVector(R, qn);
        return new double[]{
            R[0]*a[0] + R[1]*a[1] + R[2]*a[2],
            R[3]*a[0] + R[4]*a[1] + R[5]*a[2],
            R[6]*a[0] + R[7]*a[1] + R[8]*a[2]
        };
    }

    private static double sq(double x) { return x * x; }
}
