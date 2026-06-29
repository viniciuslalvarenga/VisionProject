package com.example.visionproject.vio.viewmodel;

import android.app.Application;
import android.os.SystemClock;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.vio.VioState;
import com.example.visionproject.vio.factory.KeyframeCriteriaFactory;
import com.example.visionproject.vio.model.FrameSample;
import com.example.visionproject.vio.model.KeyframePair;
import com.example.visionproject.vio.pipeline.ImuNavigator;
import com.example.visionproject.vio.pipeline.KeyframeSelector;
import com.example.visionproject.vio.repository.KeyframePairRepository;
import com.example.visionproject.vio.repository.SyncedDataRepository;
import com.example.visionproject.vio.repository.VioCsvLogger;

import org.opencv.core.Mat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VioCaptureViewModel extends AndroidViewModel {
    private static final String TAG = "VioCaptureViewModel";

    private final MutableLiveData<VioState> state = new MutableLiveData<>(VioState.IDLE);
    private final MutableLiveData<Double>   baselineRatio = new MutableLiveData<>(0.0);
    private final MutableLiveData<Double>   parallaxRatio = new MutableLiveData<>(0.0);
    private final MutableLiveData<String>   statusMessage = new MutableLiveData<>("");
    private final MutableLiveData<Long>     clockOffsetMs = new MutableLiveData<>(0L);

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ImuNavigator imuNavigator = ImuNavigator.getInstance();
    private final KeyframeSelector keyframeSelector;

    private Mat K, distCoeffs;
    private volatile float[] currentRotation = {0, 0, 0, 1};
    private volatile long lastSensorTns = 0;

    public VioCaptureViewModel(@NonNull Application app) {
        super(app);
        CalibrationResult cr = CalibrationJsonStore.load(app);
        if (cr != null) {
            K = cr.getCameraMatrix();
            distCoeffs = cr.getDistCoeffs();
        }
        keyframeSelector = new KeyframeSelector(KeyframeCriteriaFactory.balanced());
        imuNavigator.reset();
    }

    public LiveData<VioState>  getState()         { return state; }
    public LiveData<Double>    getBaselineRatio()  { return baselineRatio; }
    public LiveData<Double>    getParallaxRatio()  { return parallaxRatio; }
    public LiveData<String>    getStatusMessage()  { return statusMessage; }
    public LiveData<Long>      getClockOffsetMs()  { return clockOffsetMs; }

    public void startCapture() {
        imuNavigator.reset();
        keyframeSelector.resetReference(null);
        state.postValue(VioState.CAPTURING);
        statusMessage.postValue("Capturando... mova a câmera lateralmente.");
        VioCsvLogger.getInstance().log("IMU_STARTED", "capture started");
    }

    public void pauseCapture() {
        state.postValue(VioState.IDLE);
        VioCsvLogger.getInstance().log("IMU_STOPPED", "capture paused");
    }

    public void onAccelSample(long tNs, float[] values) {
        lastSensorTns = tNs;
        imuNavigator.onAccel(tNs, values, currentRotation);
        SyncedDataRepository.getInstance().addImuSample(
                new com.example.visionproject.vio.model.ImuSample(tNs, values,
                        android.hardware.Sensor.TYPE_LINEAR_ACCELERATION));
    }

    public void onRotationSample(long tNs, float[] values) {
        lastSensorTns = tNs;
        currentRotation = values.clone();
    }

    public void onNewFrame(long tFrameNs, Mat rgb) {
        if (state.getValue() != VioState.CAPTURING) return;
        if (K == null) {
            statusMessage.postValue("Erro: Calibração não carregada.");
            return;
        }

        FrameSample frame = new FrameSample(tFrameNs, rgb);
        SyncedDataRepository.getInstance().addFrame(frame);

        executor.execute(() -> {
            // Update readiness bar
            double bRatio = keyframeSelector.getBaselineRatio(imuNavigator, tFrameNs);
            baselineRatio.postValue(Math.min(bRatio, 1.5));

            KeyframePair pair = keyframeSelector.tryPair(frame, imuNavigator, K, distCoeffs);
            if (pair != null) {
                KeyframePairRepository.getInstance().setPair(pair);
                state.postValue(VioState.KEYFRAME_READY);
                statusMessage.postValue(String.format(java.util.Locale.US,
                        "Par aceito! Baseline=%.3fm Matches=%d",
                        pair.baseline_m, pair.matches.good.size()));
                VioCsvLogger.getInstance().logDetailed(
                        "KEYFRAME_ACCEPTED", null, tFrameNs, null,
                        null, null, null, null, null, null, null, null, null, null,
                        pair.baseline_m, pair.parallax_px, pair.matches.good.size(),
                        null, null, null, null, null, null, null, null);
            }
        });
    }

    public void diagnoseClockOffset() {
        long elapsedNs = SystemClock.elapsedRealtimeNanos();
        if (lastSensorTns == 0) return;
        long offsetMs = (elapsedNs - lastSensorTns) / 1_000_000;
        clockOffsetMs.postValue(offsetMs);
        String note = "offset_ms=" + offsetMs;
        if (Math.abs(offsetMs) > 50) note += " AVISO: offset > 50ms!";
        VioCsvLogger.getInstance().log("CLOCK_BASE_CHECK", note);
        Log.d(TAG, "Clock offset: " + offsetMs + " ms");
    }

    public void resetReference() {
        FrameSample latest = SyncedDataRepository.getInstance().getLatestFrame();
        if (latest != null) keyframeSelector.resetReference(latest);
        state.postValue(VioState.CAPTURING);
        baselineRatio.postValue(0.0);
        parallaxRatio.postValue(0.0);
        statusMessage.postValue("Referência resetada. Mova a câmera.");
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        executor.shutdown();
    }
}
