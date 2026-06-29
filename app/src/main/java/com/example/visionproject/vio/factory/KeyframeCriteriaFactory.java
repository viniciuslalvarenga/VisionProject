package com.example.visionproject.vio.factory;

import com.example.visionproject.vio.strategy.BaselineParallaxStrategy;
import com.example.visionproject.vio.strategy.KeyframeCriteria;

public class KeyframeCriteriaFactory {
    public static KeyframeCriteria strict()   { return new BaselineParallaxStrategy(0.10, 60, 100); }
    public static KeyframeCriteria balanced() { return new BaselineParallaxStrategy(0.05, 50,  80); }
    public static KeyframeCriteria loose()    { return new BaselineParallaxStrategy(0.03, 30,  50); }
}
