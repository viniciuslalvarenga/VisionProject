package com.example.visionproject.vio.viewmodel;

import android.app.Application;
import android.hardware.SensorManager;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.vio.model.Pose;
import com.example.visionproject.vio.pipeline.ImuNavigator;
import com.example.visionproject.vio.repository.VioCsvLogger;
import com.example.visionproject.vio.strategy.ZuptStrategy;

import java.util.ArrayList;
import java.util.List;

public class VioImuTestViewModel extends AndroidViewModel {
    private final MutableLiveData<Boolean>    recording   = new MutableLiveData<>(false);
    private final MutableLiveData<String>     result      = new MutableLiveData<>("");
    private final MutableLiveData<List<Pose>> trajectory1 = new MutableLiveData<>();
    private final MutableLiveData<List<Pose>> trajectory2 = new MutableLiveData<>();
    private final MutableLiveData<List<Float>> err1       = new MutableLiveData<>();
    private final MutableLiveData<List<Float>> err2       = new MutableLiveData<>();

    // Navigator WITH ZUPT
    private final ImuNavigator navWithZupt = new ImuNavigator(new ZuptStrategy.DefaultZuptStrategy());
    // Navigator WITHOUT ZUPT (never fires)
    private final ImuNavigator navNoZupt   = new ImuNavigator((mag, dt, acc) -> false);

    private volatile float[] currentRotation = {0, 0, 0, 1};

    public VioImuTestViewModel(@NonNull Application app) { super(app); }

    public LiveData<Boolean>    isRecording()     { return recording; }
    public LiveData<String>     getResult()        { return result; }
    public LiveData<List<Pose>> getTraj1()         { return trajectory1; }
    public LiveData<List<Pose>> getTraj2()         { return trajectory2; }
    public LiveData<List<Float>> getErrors1()      { return err1; }
    public LiveData<List<Float>> getErrors2()      { return err2; }

    public void startRecording() {
        navWithZupt.reset();
        navNoZupt.reset();
        recording.setValue(true);
        result.setValue("Gravando... mova 1m para frente e volte.");
        VioCsvLogger.getInstance().log("IMU_DRIFT_TEST", "started");
    }

    public void onAccelSample(long tNs, float[] values) {
        if (Boolean.TRUE.equals(recording.getValue())) {
            navWithZupt.onAccel(tNs, values, currentRotation);
            navNoZupt.onAccel(tNs, values, currentRotation);
        }
    }

    public void onRotation(float[] values) {
        currentRotation = values.clone();
    }

    public void stopRecording() {
        recording.setValue(false);

        List<Pose> h1 = navWithZupt.getHistoryCopy();
        List<Pose> h2 = navNoZupt.getHistoryCopy();

        trajectory1.setValue(h1);
        trajectory2.setValue(h2);

        List<Float> e1 = computeErrors(h1);
        List<Float> e2 = computeErrors(h2);
        err1.setValue(e1);
        err2.setValue(e2);

        double finalErrWithZupt = e1.isEmpty() ? 0 : e1.get(e1.size()-1);
        double finalErrNoZupt   = e2.isEmpty() ? 0 : e2.get(e2.size()-1);

        String msg = String.format(java.util.Locale.US,
                "Erro final COM ZUPT: %.3fm | SEM ZUPT: %.3fm",
                finalErrWithZupt, finalErrNoZupt);
        result.setValue(msg);

        VioCsvLogger.getInstance().logDetailed("IMU_DRIFT_TEST",
                null, null, null,
                finalErrWithZupt, finalErrNoZupt, null,
                null, null, null, null, null, null, null,
                null, null, null, null, null, null, null, null, null, null,
                "err_no_zupt="+String.format(java.util.Locale.US,"%.3f",finalErrNoZupt)
                + " err_with_zupt="+String.format(java.util.Locale.US,"%.3f",finalErrWithZupt));
    }

    private List<Float> computeErrors(List<Pose> history) {
        List<Float> errors = new ArrayList<>();
        if (history.isEmpty()) return errors;
        // Origin is at first pose
        double[] origin = history.get(0).p;
        for (Pose p : history) {
            double err = Math.sqrt(
                    sq(p.p[0] - origin[0]) +
                    sq(p.p[1] - origin[1]) +
                    sq(p.p[2] - origin[2]));
            errors.add((float) err);
        }
        return errors;
    }

    private static double sq(double x) { return x * x; }
}
