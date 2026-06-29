package com.example.visionproject.vio.pipeline;

public class VioFusion {
    private static final float ALPHA_DEFAULT = 0.05f;

    private float alpha;
    private double[] fusedP = {0, 0, 0};
    private float[] fusedQ = {0, 0, 0, 1};

    public VioFusion() { this.alpha = ALPHA_DEFAULT; }
    public VioFusion(float alpha) { this.alpha = alpha; }

    public void fusePose(double[] poseVisual, float[] qVisual,
                         double[] poseIMU, float[] qIMU) {
        for (int i = 0; i < 3; i++) {
            fusedP[i] = alpha * poseVisual[i] + (1 - alpha) * poseIMU[i];
        }
        fusedQ = slerp(qIMU, qVisual, alpha);
        ImuNavigator.getInstance().resetWithPose(fusedP, fusedQ);
    }

    public double[] getFusedPosition()    { return fusedP.clone(); }
    public float[]  getFusedOrientation() { return fusedQ.clone(); }
    public float    getAlpha()            { return alpha; }
    public void     setAlpha(float a)     { this.alpha = a; }

    private static float[] slerp(float[] q0, float[] q1, float t) {
        float dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];
        float[] q1c = q1.clone();
        if (dot < 0) {
            for (int i = 0; i < 4; i++) q1c[i] = -q1c[i];
            dot = -dot;
        }
        if (dot > 0.9995f) {
            float[] result = new float[4];
            for (int i = 0; i < 4; i++) result[i] = q0[i] + t * (q1c[i] - q0[i]);
            return normalize(result);
        }
        double theta0 = Math.acos(dot);
        double theta = theta0 * t;
        double sinTheta = Math.sin(theta);
        double sinTheta0 = Math.sin(theta0);
        float s0 = (float)(Math.cos(theta) - dot * sinTheta / sinTheta0);
        float s1 = (float)(sinTheta / sinTheta0);
        float[] result = new float[4];
        for (int i = 0; i < 4; i++) result[i] = s0 * q0[i] + s1 * q1c[i];
        return result;
    }

    private static float[] normalize(float[] q) {
        float norm = (float) Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        if (norm < 1e-6f) return new float[]{0, 0, 0, 1};
        return new float[]{q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm};
    }
}
