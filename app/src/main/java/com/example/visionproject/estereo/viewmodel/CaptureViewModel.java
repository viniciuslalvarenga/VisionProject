package com.example.visionproject.estereo.viewmodel;

import android.app.Application;
import android.content.SharedPreferences;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import com.example.visionproject.estereo.EstereoState;
import com.example.visionproject.estereo.model.StereoPair;
import com.example.visionproject.estereo.repository.StereoCsvLogger;
import com.example.visionproject.estereo.repository.StereoPairRepository;
import java.io.File;
import java.util.UUID;

public class CaptureViewModel extends AndroidViewModel {
    private static final String PREFS_NAME = "estereo_prefs";
    private static final String KEY_BASELINE = "baseline_mm";
    private static final float DEFAULT_BASELINE = 60.0f;
    private static final float MIN_BASELINE = 20.0f;
    private static final float MAX_BASELINE = 1000.0f;

    private final MutableLiveData<EstereoState> state = new MutableLiveData<>(EstereoState.AWAIT_LEFT);
    private final MutableLiveData<File> leftFile = new MutableLiveData<>();
    private final MutableLiveData<File> rightFile = new MutableLiveData<>();
    private final MutableLiveData<Float> baselineMm;
    private final String sessionId;
    private final SharedPreferences prefs;

    public CaptureViewModel(@NonNull Application application) {
        super(application);
        this.sessionId = UUID.randomUUID().toString().substring(0, 8);
        this.prefs = application.getSharedPreferences(PREFS_NAME, 0);
        float saved = prefs.getFloat(KEY_BASELINE, DEFAULT_BASELINE);
        this.baselineMm = new MutableLiveData<>(saved);
        StereoCsvLogger.getInstance().logDetailed("BASELINE_LOADED", null, null, (int) saved,
                null, null, null, null, null, null, null, null, null,
                null, null, null, null, "baseline carregada de SharedPreferences");
    }

    public LiveData<EstereoState> getState() { return state; }
    public LiveData<File> getLeftFile() { return leftFile; }
    public LiveData<File> getRightFile() { return rightFile; }
    public LiveData<Float> getBaselineMm() { return baselineMm; }

    public void onImageSaved(String side, File file) {
        if ("L".equals(side)) {
            leftFile.postValue(file);
            state.postValue(EstereoState.AWAIT_RIGHT);
        } else if ("R".equals(side)) {
            rightFile.postValue(file);
            state.postValue(EstereoState.PAIR_READY);
        }
    }

    public void setBaselineMm(float value) {
        if (value < MIN_BASELINE || value > MAX_BASELINE) return;
        baselineMm.setValue(value);
        prefs.edit().putFloat(KEY_BASELINE, value).apply();
        StereoCsvLogger.getInstance().logDetailed("BASELINE_SET", null, null, (int) value,
                null, null, null, null, null, null, null, null, null,
                null, null, null, null, "baseline editada pelo usuario");
    }

    public void clear() {
        leftFile.setValue(null);
        rightFile.setValue(null);
        state.setValue(EstereoState.AWAIT_LEFT);
        StereoPairRepository.getInstance().clear();
    }

    public void finalizePair() {
        if (leftFile.getValue() != null && rightFile.getValue() != null) {
            StereoPair pair = new StereoPair(leftFile.getValue(), rightFile.getValue(),
                    baselineMm.getValue(), sessionId);
            StereoPairRepository.getInstance().setCurrentPair(pair);
        }
    }
}
