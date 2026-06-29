package com.example.visionproject.vio.sensor;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SensorLogger implements SensorEventListener {

    public static class Sample {
        public final long tNs;
        public final float[] v;
        public final int type;

        public Sample(long tNs, float[] v, int type) {
            this.tNs = tNs;
            this.v = v.clone();
            this.type = type;
        }
    }

    public interface Listener {
        void onAccel(Sample s);
        void onGyro(Sample s);
        void onRotation(Sample s);
    }

    private final List<Sample> accel = Collections.synchronizedList(new ArrayList<>());
    private final List<Sample> gyro  = Collections.synchronizedList(new ArrayList<>());
    private final List<Sample> rot   = Collections.synchronizedList(new ArrayList<>());

    private Listener listener;

    public void setListener(Listener l) { this.listener = l; }

    public void start(SensorManager sm) {
        register(sm, Sensor.TYPE_LINEAR_ACCELERATION);
        register(sm, Sensor.TYPE_GYROSCOPE);
        register(sm, Sensor.TYPE_ROTATION_VECTOR);
    }

    public void stop(SensorManager sm) {
        sm.unregisterListener(this);
    }

    private void register(SensorManager sm, int type) {
        Sensor s = sm.getDefaultSensor(type);
        if (s != null) sm.registerListener(this, s, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    public void onSensorChanged(SensorEvent e) {
        Sample s = new Sample(e.timestamp, e.values, e.sensor.getType());
        switch (e.sensor.getType()) {
            case Sensor.TYPE_LINEAR_ACCELERATION:
                accel.add(s);
                if (listener != null) listener.onAccel(s);
                break;
            case Sensor.TYPE_GYROSCOPE:
                gyro.add(s);
                if (listener != null) listener.onGyro(s);
                break;
            case Sensor.TYPE_ROTATION_VECTOR:
                rot.add(s);
                if (listener != null) listener.onRotation(s);
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor s, int a) {}

    public List<Sample> getAccel()    { return new ArrayList<>(accel); }
    public List<Sample> getGyro()     { return new ArrayList<>(gyro); }
    public List<Sample> getRotation() { return new ArrayList<>(rot); }

    public void clear() { accel.clear(); gyro.clear(); rot.clear(); }

    public int getAccelCount() { return accel.size(); }
    public int getGyroCount()  { return gyro.size(); }
    public int getRotCount()   { return rot.size(); }
}
