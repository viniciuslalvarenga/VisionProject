package com.example.visionproject.modelocamera;

import android.app.Application;
import android.graphics.Bitmap;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.visionproject.R;
import com.example.visionproject.modelocamera.logger.CalibrationSessionLogger;
import com.example.visionproject.modelocamera.logger.event.CalibrationLoadedEvent;
import com.example.visionproject.modelocamera.logger.event.EpipolarPointAddedEvent;
import com.example.visionproject.modelocamera.logger.event.ImageCapturedEvent;
import com.example.visionproject.modelocamera.logger.event.ImageUndistortedEvent;
import com.example.visionproject.modelocamera.logger.event.PointUndistortedEvent;

import com.example.visionproject.modelocamera.model.CameraIntrinsics;
import com.example.visionproject.modelocamera.model.DistortionCoefficients;
import com.example.visionproject.modelocamera.model.EpipolarPoint;
import com.example.visionproject.modelocamera.repository.CalibrationRepository;
import com.example.visionproject.modelocamera.strategy.UndistortStrategy;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

/**
 * ViewModel para a Activity ModeloCamera, orquestrando a lógica de negócio e o estado da UI.
 */
public class ModeloCameraViewModel extends AndroidViewModel {

    private final MutableLiveData<CameraIntrinsics> intrinsics = new MutableLiveData<>();
    private final MutableLiveData<DistortionCoefficients> distortion = new MutableLiveData<>();
    private final MutableLiveData<Bitmap> originalImage = new MutableLiveData<>();
    private final MutableLiveData<Bitmap> correctedImage = new MutableLiveData<>();
    private final MutableLiveData<List<EpipolarPoint>> epipolarPoints = new MutableLiveData<>(new ArrayList<>());
    private final MutableLiveData<String> statusMessage = new MutableLiveData<>();

    private final UndistortStrategy<Mat, Mat> imageStrategy;
    private final UndistortStrategy<Point, Point> pointStrategy;
    private final CalibrationRepository repository;
    private final CalibrationSessionLogger logger = CalibrationSessionLogger.getInstance();

    public ModeloCameraViewModel(@NonNull Application application,
                                 UndistortStrategy<Mat, Mat> imgStrategy,
                                 UndistortStrategy<Point, Point> ptStrategy,
                                 CalibrationRepository repo) {
        super(application);
        this.imageStrategy = imgStrategy;
        this.pointStrategy = ptStrategy;
        this.repository = repo;
        
        logger.startSession(application);
        loadDefaultCalibration();
    }

    public void loadDefaultCalibration() {
        intrinsics.setValue(repository.getIntrinsics());
        distortion.setValue(repository.getDistortion());
        statusMessage.setValue(getApplication().getString(R.string.mc_status_calibration_loaded));
        
        logger.log(new CalibrationLoadedEvent(repository.getIntrinsics(), repository.getDistortion()));
    }

    public void onImageCaptured(Mat originalMat) {
        if (originalMat == null || originalMat.empty()) {
            statusMessage.setValue(getApplication().getString(R.string.mc_error_empty_image));
            return;
        }

        statusMessage.setValue(getApplication().getString(R.string.mc_status_processing));

        // Converte original para Bitmap
        Bitmap originalBmp = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(originalMat, originalBmp);
        originalImage.postValue(originalBmp);
        
        logger.log(new ImageCapturedEvent("captured_image.png", originalMat.cols(), originalMat.rows(), "CAMERA"));

        // Aplica correção
        long startTime = System.currentTimeMillis();
        Mat correctedMat = imageStrategy.undistort(originalMat, intrinsics.getValue(), distortion.getValue());
        long elapsed = System.currentTimeMillis() - startTime;
        
        if (!correctedMat.empty()) {
            Bitmap correctedBmp = Bitmap.createBitmap(correctedMat.cols(), correctedMat.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(correctedMat, correctedBmp);
            correctedImage.postValue(correctedBmp);
            statusMessage.postValue(getApplication().getString(R.string.mc_status_success));
            
            logger.log(new ImageUndistortedEvent("captured_image.png", "corrected_image.png", correctedMat.cols(), correctedMat.rows(), elapsed));
        } else {
            statusMessage.postValue(getApplication().getString(R.string.mc_status_error));
        }
        
        correctedMat.release();
    }

    public void onPointSelected(Point p) {
        if (p == null || intrinsics.getValue() == null) return;
        
        Point corrected = pointStrategy.undistort(p, intrinsics.getValue(), distortion.getValue());
        
        // Calcula distância do centro (principal point)
        double dx = p.x - intrinsics.getValue().getCx();
        double dy = p.y - intrinsics.getValue().getCy();
        double dist = Math.sqrt(dx*dx + dy*dy);
        
        logger.log(new PointUndistortedEvent(p, corrected, dist));
    }

    public void onEpipolarPointAdded(Point p, int idx, int color, String label) {
        List<EpipolarPoint> current = epipolarPoints.getValue();
        if (current == null) current = new ArrayList<>();
        
        current.add(new EpipolarPoint(p, color, idx));
        epipolarPoints.setValue(current);
        
        logger.log(new EpipolarPointAddedEvent(p, idx, color, label));
    }

    public void resetEpipolar() {
        epipolarPoints.setValue(new ArrayList<>());
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        logger.endSession(getApplication(), "DESTROY");
    }

    // Getters para LiveData
    public LiveData<CameraIntrinsics> getIntrinsics() { return intrinsics; }
    public LiveData<DistortionCoefficients> getDistortion() { return distortion; }
    public LiveData<Bitmap> getOriginalImage() { return originalImage; }
    public LiveData<Bitmap> getCorrectedImage() { return correctedImage; }
    public LiveData<List<EpipolarPoint>> getEpipolarPoints() { return epipolarPoints; }
    public LiveData<String> getStatusMessage() { return statusMessage; }
}
