package com.example.visionproject.vio.model;

import com.example.visionproject.vio.strategy.OrbMatcherStrategy;

public class KeyframePair {
    public final FrameSample a;
    public final FrameSample b;
    public final double baseline_m;
    public final double parallax_px;
    public final OrbMatcherStrategy.MatchResult matches;

    public KeyframePair(FrameSample a, FrameSample b, double baseline_m, double parallax_px,
                        OrbMatcherStrategy.MatchResult matches) {
        this.a = a;
        this.b = b;
        this.baseline_m = baseline_m;
        this.parallax_px = parallax_px;
        this.matches = matches;
    }
}
