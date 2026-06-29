package com.example.visionproject.vio.strategy;

public interface ZuptStrategy {
    boolean isAtRest(double magnitude, double dt, double restAccum);

    class DefaultZuptStrategy implements ZuptStrategy {
        private static final double EPS = 0.08;
        private static final double REST_T = 0.30;

        @Override
        public boolean isAtRest(double magnitude, double dt, double restAccum) {
            return magnitude < EPS && restAccum >= REST_T;
        }
    }
}
