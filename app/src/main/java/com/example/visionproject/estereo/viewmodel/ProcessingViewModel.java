package com.example.visionproject.estereo.viewmodel;

import android.app.Application;
import android.graphics.Bitmap;
import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.estereo.EstereoState;
import com.example.visionproject.estereo.model.StereoPair;
import com.example.visionproject.estereo.pipeline.StereoPipeline;
import com.example.visionproject.estereo.repository.StereoCsvLogger;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ProcessingViewModel extends AndroidViewModel {
    private final MutableLiveData<EstereoState> state = new MutableLiveData<>(EstereoState.IDLE);
    private final MutableLiveData<Bitmap> currentArtifact = new MutableLiveData<>();
    private final MutableLiveData<String> statusMessage = new MutableLiveData<>();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final StereoPipeline pipeline = new StereoPipeline();

    public ProcessingViewModel(@NonNull Application application) {
        super(application);
    }

    public LiveData<EstereoState> getState() { return state; }
    public LiveData<Bitmap> getCurrentArtifact() { return currentArtifact; }
    public LiveData<String> getStatusMessage() { return statusMessage; }

    public void processPair(StereoPair pair) {
        if (pair == null) {
            statusMessage.setValue("Erro: Par estereo nao encontrado.");
            state.setValue(EstereoState.ERROR);
            return;
        }

        CalibrationResult calib = CalibrationJsonStore.load(getApplication());
        if (calib == null) {
            statusMessage.setValue("Erro: Calibracao nao encontrada. Execute o Modulo 3.");
            state.setValue(EstereoState.ERROR);
            return;
        }
        Mat K = calib.getCameraMatrix();
        if (K == null || K.rows() < 1 || K.cols() < 1) {
            statusMessage.setValue("Erro: Matriz K invalida (null ou dimensoes incorretas).");
            state.setValue(EstereoState.ERROR);
            return;
        }
        double fx = K.get(0, 0)[0];
        StereoCsvLogger.getInstance().logDetailed("CALIBRATION_LOADED",
                null, null, (int) pair.baselineMm,
                null, null, null, null, null, null, null, null, fx,
                null, null, null, null, "K e D carregados do calibration.json");

        state.postValue(EstereoState.PROCESSING);
        statusMessage.postValue("Iniciando processamento...");

        final String sceneId = "scene_" + System.currentTimeMillis();
        executor.execute(() -> {
            pipeline.run(getApplication(), pair.leftFile, pair.rightFile,
                    calib.getCameraMatrix(), calib.getDistCoeffs(),
                    sceneId, (int) pair.baselineMm,
                    new StereoPipeline.PipelineCallback() {
                        @Override
                        public void onStepDone(String stepName, Mat artifact) {
                            statusMessage.postValue("Etapa: " + stepName);
                            if (artifact != null && !artifact.empty()) {
                                try {
                                    Bitmap bmp = Bitmap.createBitmap(artifact.cols(), artifact.rows(),
                                            Bitmap.Config.ARGB_8888);
                                    Utils.matToBitmap(artifact, bmp);
                                    currentArtifact.postValue(bmp);
                                } catch (Exception ignored) {}
                            }
                        }

                        @Override
                        public void onError(String message) {
                            statusMessage.postValue("Erro: " + message);
                            state.postValue(EstereoState.ERROR);
                        }
                    });

            if (state.getValue() != EstereoState.ERROR) {
                statusMessage.postValue("Processamento completo!");
                state.postValue(EstereoState.DONE);
            }
        });
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        executor.shutdown();
    }
}
