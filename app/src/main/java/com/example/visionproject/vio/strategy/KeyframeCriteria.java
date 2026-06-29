package com.example.visionproject.vio.strategy;

public interface KeyframeCriteria {
    boolean isReady(double baseline_m, double parallax_px, int matches);
    double getMinBaseline();
    double getMinParallax();
    int getMinMatches();
}
