package com.example.visionproject.shared;

import android.os.Build;

/**
 * Fornece informações básicas do dispositivo para logging.
 */
public class DeviceInfoProvider {
    public static String getDeviceModel() {
        return Build.MANUFACTURER + " " + Build.MODEL;
    }

    public static String getAndroidVersion() {
        return Build.VERSION.RELEASE;
    }
}
