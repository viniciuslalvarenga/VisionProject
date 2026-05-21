package com.example.visionproject.calibracao;

import android.app.Application;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.calibracao.model.CalibrationFrame;
import com.example.visionproject.calibracao.model.CalibrationResult;
import com.example.visionproject.calibracao.model.CalibrationState;
import com.example.visionproject.calibracao.model.ChessboardSpec;
import com.example.visionproject.calibracao.model.PatternSpec;
import com.example.visionproject.calibracao.repository.CalibrationCsvLogger;
import com.example.visionproject.calibracao.repository.CalibrationFramesRepository;
import com.example.visionproject.calibracao.repository.CalibrationJsonStore;
import com.example.visionproject.calibracao.strategy.BlurDetectionStrategy;
import com.example.visionproject.calibracao.strategy.ChessboardDetectionStrategy;
import com.example.visionproject.calibracao.strategy.PatternDetectionStrategy;
import com.example.visionproject.calibracao.strategy.PoseStabilityStrategy;
import com.example.visionproject.calibracao.pipeline.ZhangCalibrationPipeline;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CalibrationViewModel extends AndroidViewModel {
    private static final String TAG = "CalibrationViewModel";

    private final MutableLiveData<CalibrationState> state = new MutableLiveData<>(CalibrationState.IDLE);
    private final MutableLiveData<Integer> framesCollectedCount = new MutableLiveData<>(0);
    private final MutableLiveData<PatternDetectionStrategy.DetectionResult> lastDetection = new MutableLiveData<>();
    private final MutableLiveData<Double> lastBlurScore = new MutableLiveData<>(0.0);
    private final MutableLiveData<Boolean> autoCaptureEnabled = new MutableLiveData<>(true);
    private final MutableLiveData<CalibrationResult> result = new MutableLiveData<>();
    private final MutableLiveData<String> userMessage = new MutableLiveData<>();
    private final MutableLiveData<android.util.Pair<android.graphics.Bitmap, android.graphics.Bitmap>> undistortComparison = new MutableLiveData<>();

    private final PatternSpec patternSpec = new ChessboardSpec(9, 6, 25.0f);
    private final PatternDetectionStrategy detector = new ChessboardDetectionStrategy();
    private final BlurDetectionStrategy blurStrategy = new BlurDetectionStrategy();
    private final PoseStabilityStrategy stabilityStrategy = new PoseStabilityStrategy();
    private final ZhangCalibrationPipeline pipeline = new ZhangCalibrationPipeline();
    private final CalibrationFramesRepository framesRepo = CalibrationFramesRepository.getInstance();

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Mat mGray = new Mat();
    private Size lastImageSize = new Size(0, 0);
    private long lastDetectedLogMs = 0;

    public CalibrationViewModel(@NonNull Application application) {
        super(application);
        framesCollectedCount.setValue(framesRepo.size());
    }

    public LiveData<CalibrationState> getState() { return state; }
    public LiveData<Integer> getFramesCollectedCount() { return framesCollectedCount; }
    public LiveData<PatternDetectionStrategy.DetectionResult> getLastDetection() { return lastDetection; }
    public LiveData<Double> getLastBlurScore() { return lastBlurScore; }
    public LiveData<Boolean> getAutoCaptureEnabled() { return autoCaptureEnabled; }
    public LiveData<CalibrationResult> getResult() { return result; }
    public LiveData<String> getUserMessage() { return userMessage; }
    public LiveData<android.util.Pair<android.graphics.Bitmap, android.graphics.Bitmap>> getUndistortComparison() { return undistortComparison; }

    public Size getLastImageSize() { return lastImageSize; }

    public void onPreviewFrame(Mat rgba) {
        if (state.getValue() == CalibrationState.CALIBRATING || state.getValue() == CalibrationState.DONE) {
            return;
        }

        lastImageSize = new Size(rgba.cols(), rgba.rows());
        Imgproc.cvtColor(rgba, mGray, Imgproc.COLOR_RGBA2GRAY);
        
        PatternDetectionStrategy.DetectionResult detection = detector.detect(mGray, patternSpec);
        lastDetection.postValue(detection);

        if (detection.found) {
            double blurScore = blurStrategy.getScore(mGray);
            lastBlurScore.postValue(blurScore);

            // Rate-limited log for detected frames
            long now = System.currentTimeMillis();
            if (lastDetectedLogMs + 1000 < now) {
                CalibrationCsvLogger.getInstance().logFrameDetected((int) detection.corners.total(), blurScore);
                lastDetectedLogMs = now;
            }

            boolean isSharp = blurScore >= 100.0;
            boolean isStable = stabilityStrategy.checkStability(detection.corners);

            if (isSharp && isStable) {
                state.postValue(CalibrationState.POSE_STABLE);
                if (Boolean.TRUE.equals(autoCaptureEnabled.getValue())) {
                    double stability = stabilityStrategy.getCurrentStabilityScore();
                    captureFrame(rgba, detection.corners, blurScore, stability);
                }
            } else {
                state.postValue(CalibrationState.DETECTING);
                if (isStable && !isSharp && Boolean.TRUE.equals(autoCaptureEnabled.getValue())) {
                    // Log rejection due to blur if stable but blurry
                    CalibrationCsvLogger.getInstance().logFrameRejected("blur", blurScore);
                }
            }
        } else {
            state.postValue(CalibrationState.IDLE);
            stabilityStrategy.reset();
        }
    }

    public void captureFrame(Mat rgba) {
        PatternDetectionStrategy.DetectionResult detection = lastDetection.getValue();
        if (detection != null && detection.found) {
            double blurScore = lastBlurScore.getValue() != null ? lastBlurScore.getValue() : 0.0;
            double stability = stabilityStrategy.getCurrentStabilityScore();
            captureFrame(rgba, detection.corners, blurScore, stability);
            userMessage.postValue("Frame " + framesRepo.size() + " capturado e salvo.");
        } else {
            userMessage.postValue("Padrão não detectado! Centralize o tabuleiro.");
        }
    }

    private synchronized void captureFrame(Mat rgba, MatOfPoint2f corners, double blur, double stability) {
        MatOfPoint2f cornersClone = new MatOfPoint2f();
        corners.copyTo(cornersClone);

        int frameIdx = framesRepo.size();
        CalibrationFrame frame = new CalibrationFrame(
                frameIdx,
                System.currentTimeMillis(),
                cornersClone,
                (int) corners.total(),
                blur,
                stability
        );
        
        framesRepo.addFrame(frame);
        saveCapturedImage(rgba, frameIdx); // Salva a imagem no disco

        CalibrationCsvLogger.getInstance().logFrameCaptured(frame, frameIdx);
        framesCollectedCount.postValue(framesRepo.size());
        state.postValue(CalibrationState.CAPTURED);
        
        if (framesRepo.size() >= 15) {
            state.postValue(CalibrationState.READY_TO_CALIBRATE);
        }
    }

    private void saveCapturedImage(Mat rgba, int index) {
        try {
            File dir = getVisionProjectDir("Captures");
            if (dir == null) return;
            
            String fileName = "cal_img_" + index + ".jpg";
            File file = new File(dir, fileName);
            
            Mat bgr = new Mat();
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
            org.opencv.imgcodecs.Imgcodecs.imwrite(file.getAbsolutePath(), bgr);
            bgr.release();
            Log.d(TAG, "Imagem salva: " + file.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Erro ao salvar imagem capturada", e);
        }
    }

    private java.io.File getVisionProjectDir(String subDir) {
        java.io.File baseDir;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            // No Android 10+ (Scoped Storage), usamos o diretório da aplicação para evitar problemas de permissão
            baseDir = new java.io.File(getApplication().getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES), "VisionProject");
        } else {
            baseDir = new java.io.File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_PICTURES), "VisionProject");
        }
        
        java.io.File targetDir = subDir != null ? new java.io.File(baseDir, subDir) : baseDir;
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            Log.e(TAG, "Não foi possível criar diretório: " + targetDir.getAbsolutePath());
            return null;
        }
        return targetDir;
    }

    public void deleteFrame(int index) {
        framesRepo.removeAt(index);
        framesCollectedCount.postValue(framesRepo.size());
        CalibrationCsvLogger.getInstance().logFrameRejected("manual_delete", -1);
        
        // Tenta deletar o arquivo físico correspondente para manter sincronia
        try {
            File dir = getVisionProjectDir("Captures");
            if (dir != null) {
                File file = new File(dir, "cal_img_" + index + ".jpg");
                if (file.exists()) file.delete();
            }
        } catch (Exception e) {
            Log.e(TAG, "Erro ao deletar imagem física", e);
        }
        
        if (state.getValue() == CalibrationState.DONE) {
            state.postValue(CalibrationState.READY_TO_CALIBRATE);
            result.postValue(null); 
        } else if (framesRepo.size() < 15) {
            state.postValue(CalibrationState.IDLE);
        }
    }

    public void toggleAutoCapture() {
        autoCaptureEnabled.setValue(!Boolean.TRUE.equals(autoCaptureEnabled.getValue()));
    }

    public void clearFrames() {
        framesRepo.clear();
        framesCollectedCount.setValue(0);
        state.setValue(CalibrationState.IDLE);
        stabilityStrategy.reset();
    }

    public void runCalibration() {
        if (framesRepo.size() < 15) {
            userMessage.setValue("Erro: Colete pelo menos 15 imagens.");
            return;
        }

        if (lastImageSize.width == 0) {
            userMessage.setValue("Erro: Aguarde o preview da câmera carregar.");
            return;
        }

        userMessage.setValue("Calculando calibração... Isso pode levar alguns segundos.");
        Log.i(TAG, "Iniciando calibração com tamanho: " + lastImageSize);
        state.setValue(CalibrationState.CALIBRATING);
        
        executor.execute(() -> {
            try {
                CalibrationResult res = pipeline.run(framesRepo.getAll(), patternSpec, lastImageSize);
                if (res != null && res.getRms() > 0) {
                    result.postValue(res);
                    state.postValue(CalibrationState.DONE);
                    userMessage.postValue("Sucesso! Calibração concluída.");
                } else {
                    throw new Exception("Dados insuficientes para calcular.");
                }
            } catch (Exception e) {
                Log.e(TAG, "Erro fatal na calibração", e);
                userMessage.postValue("Erro: " + e.getMessage());
                state.postValue(CalibrationState.ERROR);
            }
        });
    }

    public void saveResult() {
        CalibrationResult res = result.getValue();
        if (res != null) {
            try {
                // Usa o novo método para obter o diretório correto
                java.io.File dir = getVisionProjectDir(null);
                String jsonPath = CalibrationJsonStore.saveToDir(res, getApplication(), dir);
                
                CalibrationCsvLogger.getInstance().logJsonExported(jsonPath);
                CalibrationCsvLogger.getInstance().logCalibrationDone(res, res.getPerImageReprojectionError());
                CalibrationCsvLogger.getInstance().saveSession(getApplication());
                userMessage.setValue("Calibração e logs salvos em: " + jsonPath);
            } catch (Exception e) {
                userMessage.setValue("Erro ao salvar: " + e.getMessage());
                Log.e(TAG, "Erro ao salvar resultado", e);
            }
        } else {
            userMessage.setValue("Não há resultado de calibração para salvar.");
        }
    }

    public void generateUndistortPreview(Mat lastRgba) {
        CalibrationResult res = result.getValue();
        if (res == null) {
            userMessage.setValue("Calibre a câmera primeiro.");
            return;
        }
        if (lastRgba == null || lastRgba.empty()) {
            userMessage.setValue("Nenhuma imagem capturada para comparar.");
            return;
        }

        userMessage.setValue("Gerando prévia antes/depois...");
        long startTime = System.currentTimeMillis();
        executor.execute(() -> {
            Mat undistorted = new Mat();
            org.opencv.calib3d.Calib3d.undistort(lastRgba, undistorted, res.getCameraMatrix(), res.getDistCoeffs());
            long elapsed = System.currentTimeMillis() - startTime;
            
            CalibrationCsvLogger.getInstance().logUndistortPreview(lastRgba.cols(), lastRgba.rows(), elapsed);

            android.graphics.Bitmap bmpOriginal = android.graphics.Bitmap.createBitmap(lastRgba.cols(), lastRgba.rows(), android.graphics.Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(lastRgba, bmpOriginal);

            android.graphics.Bitmap bmpUndistorted = android.graphics.Bitmap.createBitmap(undistorted.cols(), undistorted.rows(), android.graphics.Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(undistorted, bmpUndistorted);

            undistortComparison.postValue(new android.util.Pair<>(bmpOriginal, bmpUndistorted));
            undistorted.release();
        });
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        mGray.release();
        executor.shutdown();
    }
}
