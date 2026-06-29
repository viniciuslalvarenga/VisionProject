package com.example.visionproject.vio.strategy;

public class BaselineParallaxStrategy implements KeyframeCriteria {
    public final double sMin;
    public final double pMin;
    public final int matchMin;

    public BaselineParallaxStrategy(double sMin, double pMin, int matchMin) {
        this.sMin = sMin;
        this.pMin = pMin;
        this.matchMin = matchMin;
    }

    @Override
    public boolean isReady(double baseline_m, double parallax_px, int matches) {
        return baseline_m >= sMin && parallax_px >= pMin && matches >= matchMin;
    }

    @Override
    public double getMinBaseline() {
        return sMin;
    }

    @Override
    public double getMinParallax() {
        return pMin;
    }

    @Override
    public int getMinMatches() {
        return matchMin;
    }
}
