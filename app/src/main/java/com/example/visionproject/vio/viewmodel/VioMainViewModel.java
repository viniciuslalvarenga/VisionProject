package com.example.visionproject.vio.viewmodel;

import android.app.Application;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;

public class VioMainViewModel extends AndroidViewModel {
    private final MutableLiveData<CalibrationResult> calibration = new MutableLiveData<>();
    private final MutableLiveData<String> statusMessage = new MutableLiveData<>();

    public VioMainViewModel(@NonNull Application app) {
        super(app);
        loadCalibration();
    }

    public LiveData<CalibrationResult> getCalibration() { return calibration; }
    public LiveData<String> getStatusMessage()          { return statusMessage; }

    public void loadCalibration() {
        CalibrationResult cr = CalibrationJsonStore.load(getApplication());
        calibration.setValue(cr);
        if (cr == null) {
            statusMessage.setValue("Calibração NÃO encontrada. Execute o Módulo 3 primeiro.");
        } else {
            statusMessage.setValue(String.format(java.util.Locale.US,
                    "Calibração OK: fx=%.1f RMS=%.3f", cr.getFx(), cr.getRms()));
        }
    }
}
