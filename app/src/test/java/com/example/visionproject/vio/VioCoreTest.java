package com.example.visionproject.vio;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.example.visionproject.vio.factory.KeyframeCriteriaFactory;
import com.example.visionproject.vio.pipeline.ImuNavigator;
import com.example.visionproject.vio.strategy.BaselineParallaxStrategy;
import com.example.visionproject.vio.strategy.KeyframeCriteria;
import com.example.visionproject.vio.strategy.ZuptStrategy;

import org.junit.Test;

public class VioCoreTest {

    @Test
    public void imuNavigatorIntegratesAcceleration() {
        ImuNavigator navigator = new ImuNavigator(new ZuptStrategy() {
            @Override
            public boolean isAtRest(double magnitude, double dt, double restAccum) {
                return false;
            }
        });

        navigator.resetWithPose(new double[]{0.0, 0.0, 0.0}, new float[]{0f, 0f, 0f, 1f});
        navigator.onAccel(0L, new float[]{1f, 0f, 0f}, new float[]{0f, 0f, 0f, 1f});
        navigator.onAccel(500_000_000L, new float[]{1f, 0f, 0f}, new float[]{0f, 0f, 0f, 1f});

        Pose pose = navigator.getPoseAt(500_000_000L);
        assertTrue(pose.p[0] > 0.0);
    }

    @Test
    public void baselineCriteriaFactoryProvidesExpectedThresholds() {
        KeyframeCriteria strict = KeyframeCriteriaFactory.strict();
        KeyframeCriteria balanced = KeyframeCriteriaFactory.balanced();
        KeyframeCriteria loose = KeyframeCriteriaFactory.loose();

        assertTrue(strict instanceof BaselineParallaxStrategy);
        assertEquals(0.10, strict.getMinBaseline(), 1e-9);
        assertEquals(0.05, balanced.getMinBaseline(), 1e-9);
        assertEquals(0.03, loose.getMinBaseline(), 1e-9);
    }
}
