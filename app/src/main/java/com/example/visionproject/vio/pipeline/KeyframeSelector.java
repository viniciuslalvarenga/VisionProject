package com.example.visionproject.vio.pipeline;

import com.example.visionproject.vio.model.FrameSample;
import com.example.visionproject.vio.model.KeyframePair;
import com.example.visionproject.vio.repository.VioCsvLogger;
import com.example.visionproject.vio.strategy.KeyframeCriteria;
import com.example.visionproject.vio.strategy.OrbMatcherStrategy;

import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.List;
import java.util.Locale;

public class KeyframeSelector {
    private final KeyframeCriteria criteria;
    private final OrbMatcherStrategy orb;
    private volatile FrameSample ref;

    public KeyframeSelector(KeyframeCriteria criteria) {
        this.criteria = criteria;
        this.orb = new OrbMatcherStrategy();
    }

    public KeyframePair tryPair(FrameSample cur, ImuNavigator imu, Mat K, Mat distCoeffs) {
        // Capture volatile ref once to avoid race with resetReference(null)
        FrameSample localRef = this.ref;
        if (localRef == null) { this.ref = cur; return null; }

        double baseline = imu.distanceBetween(localRef.tNs, cur.tNs);
        if (baseline < criteria.getMinBaseline()) {
            logReject("baseline", baseline, 0, 0);
            return null;
        }

        if (localRef.rgb == null || localRef.rgb.empty() || cur.rgb == null || cur.rgb.empty()) {
            logReject("invalid_frame", baseline, 0, 0);
            return null;
        }

        OrbMatcherStrategy.MatchResult m = orb.match(localRef.rgb, cur.rgb, K, distCoeffs);
        double parallax = medianPixelShift(m);
        int matches = m.good.size();

        if (!criteria.isReady(baseline, parallax, matches)) {
            logReject(matches < criteria.getMinMatches() ? "matches" : "parallax",
                    baseline, parallax, matches);
            return null;
        }

        VioCsvLogger.getInstance().logDetailed(
                "KEYFRAME_ACCEPTED", null, null, null,
                null, null, null, null, null, null, null, null, null, null,
                baseline, parallax, matches,
                null, null, null, null, null, null, null,
                String.format(Locale.US, "baseline=%.3fm parallax=%.1fpx matches=%d", baseline, parallax, matches));

        return new KeyframePair(localRef, cur, baseline, parallax, m);
    }

    public void resetReference(FrameSample newRef) { this.ref = newRef; }

    public double getBaselineRatio(ImuNavigator imu, long curTns) {
        FrameSample localRef = this.ref;
        if (localRef == null || imu == null) return 0;
        double baseline = imu.distanceBetween(localRef.tNs, curTns);
        return baseline / criteria.getMinBaseline();
    }

    private double medianPixelShift(OrbMatcherStrategy.MatchResult m) {
        if (m.good.isEmpty()) return 0;
        List<KeyPoint> kpLList = m.kpL.toList();
        List<KeyPoint> kpRList = m.kpR.toList();
        double[] shifts = new double[m.good.size()];
        for (int i = 0; i < m.good.size(); i++) {
            DMatch d = m.good.get(i);
            Point a = kpLList.get(d.queryIdx).pt;
            Point b = kpRList.get(d.trainIdx).pt;
            shifts[i] = Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
        }
        java.util.Arrays.sort(shifts);
        return shifts[shifts.length / 2];
    }

    private void logReject(String reason, double baseline, double parallax, int matches) {
        VioCsvLogger.getInstance().logDetailed(
                "KEYFRAME_REJECTED", null, null, null,
                null, null, null, null, null, null, null, null, null, null,
                baseline, parallax, matches,
                null, null, null, null, null, null, null,
                "reason=" + reason);
    }
}
