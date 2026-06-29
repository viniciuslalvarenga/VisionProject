# ESPECIFICAÇÃO TÉCNICA — Módulo Final: Visão Estéreo SEM Suporte (VIO)
**Projeto:** VisionProject (Android, Java + OpenCV4Android + CameraX + SensorManager)
**Atividade:** ATIVIDADE FINAL — Estéreo Monocular Sem Suporte: substituindo o trilho mecânico por VIO (UFLA, Prof. Arthur de Miranda Neto)
**Pré-requisitos:** Módulos 1 (PCC), 2 (ModeloCamera), 3 (Calibração) e 4 (Estéreo) **já implementados e funcionais**

---

## 0. Princípios Não-Negociáveis (LEIA TODOS antes de codar)

1. **Zero alteração nos pacotes existentes:** `pcc/`, `modelocamera/`, `calibracao/`, `estereo/`, `shared/` continuam **intactos**. Qualquer mudança na API pública é proibida.
2. **Tudo em pacote novo:** `com.example.visionproject.vio`.
3. **Reutilizar agressivamente** o que já existe:
   - **Calibração** (K, D) → carregar de `CalibrationJsonStore.load(ctx)` (Módulo 3)
   - **ORB matching** → usar a mesma lógica do `StereoPipeline` (Módulo 4) — extrair para `vio/strategy/OrbMatcherStrategy.java` (cópia, não import — para não acoplar)
   - **SGBM** → usar `SgbmDisparityStrategy` do Módulo 4 (este pode ser importado por ser puramente algorítmico)
   - **CSV Logger pattern** → seguir mesma infra do `StereoCsvLogger` (Singleton + ExecutorService + `esc()/n()/f()`)
   - **CameraX setup** → o padrão `setupCamera()` do Módulo 4 serve como template
4. **CameraX dependências já estão no `build.gradle`** (adicionadas no Módulo 4). Não precisa re-adicionar.
5. **MVVM obrigatório** + padrões: Strategy, Singleton, Factory, Builder, Template Method, Observer (LiveData).
6. **Threading:** todas operações pesadas (E, RANSAC, SGBM, ORB) em `ExecutorService` single-thread. IMU é callback do `SensorManager` — leve, não precisa thread.
7. **Honestidade científica:** a atividade premia diagnóstico de falhas. O código deve logar **TUDO** que aconteceu (offsets de tempo, drift, rejeições de keyframe, ângulos de rotação relativos).
8. **Base de tempo comum:** `SensorEvent.timestamp` e `ImageInfo.getTimestamp()` devem estar ambos em `elapsedRealtimeNanos`. **Validar empiricamente** e logar o offset.

---

## 1. Estrutura de Pacotes (somente adições)

```
com.example.visionproject/
├── pcc/                              [JÁ EXISTE — INTACTO]
├── modelocamera/                     [JÁ EXISTE — INTACTO]
├── calibracao/                       [JÁ EXISTE — INTACTO; será CONSUMIDOR de K/D]
├── estereo/                          [JÁ EXISTE — INTACTO; pode ser referenciado]
├── shared/                           [JÁ EXISTE — REUSAR csv/]
└── vio/                              [NOVO — todo o trabalho aqui]
    ├── VioMainActivity.java          (hub: status calibração, status IMU, navegação)
    ├── VioCaptureActivity.java       (T2: PreviewView + ImageAnalysis + SensorLogger)
    ├── VioProcessingActivity.java    (T6+T7+T8: pose relativa, retificação, disparidade)
    ├── VioImuTestActivity.java       (T3: experimento de drift IMU, com gráfico erro×tempo)
    ├── VioState.java                 (enum: IDLE, CAPTURING, WAITING_KEYFRAME, KEYFRAME_READY, PROCESSING, DONE, ERROR)
    ├── model/
    │   ├── ImuSample.java            (long tNs, float[] values, int sensorType)
    │   ├── Pose.java                 (long tNs, double[] p[3], float[] q[4])
    │   ├── FrameSample.java          (long tNs, Mat rgb, ou Bitmap thumb)
    │   ├── KeyframePair.java         (FrameSample A, FrameSample B, baseline_m, parallax_px, matches)
    │   ├── RelativePoseResult.java   (Mat R_rel, Mat t_metric, int inliers, double rotAngleDeg)
    │   ├── CalibratedRectifyResult.java (rectL, rectR, R1, R2, P1, P2, Q)
    │   └── DepthEstimateResult.java  (Mat depth_xyz, double medianZ, double measuredZ_at_point)
    ├── strategy/
    │   ├── KeyframeCriteria.java     (interface)
    │   ├── BaselineParallaxStrategy.java
    │   ├── ZuptStrategy.java         (Zero-velocity Update — interface + default impl)
    │   └── OrbMatcherStrategy.java   (cópia ORB+Lowe do M4 — desacoplado)
    ├── factory/
    │   └── KeyframeCriteriaFactory.java  (presets: STRICT, BALANCED, LOOSE)
    ├── pipeline/
    │   ├── ImuNavigator.java         (T3 — integração + ZUPT, com getPoseAt(tNs))
    │   ├── KeyframeSelector.java     (T4 — orquestra: pega ref + cur, testa critérios)
    │   ├── RelativePoseEstimator.java (T6 — E + recoverPose + escala IMU)
    │   ├── CalibratedRectifier.java  (T7 — stereoRectify + initUndistortRectifyMap + remap)
    │   ├── DepthFromQ.java           (T7 — reprojectImageTo3D)
    │   └── VioFusion.java            (T8 — filtro complementar)
    ├── repository/
    │   ├── SyncedDataRepository.java (Singleton: buffer circular de frames + IMU)
    │   ├── KeyframePairRepository.java (Singleton: par escolhido para processar)
    │   └── VioCsvLogger.java         (Singleton — mesma infra)
    ├── sensor/
    │   └── SensorLogger.java         (SensorEventListener; subscribe LINEAR_ACCEL+GYRO+ROTATION_VECTOR)
    ├── ui/
    │   ├── ReadinessBarView.java     (T4 — barra de "prontidão do par" combinando baseline+parallax)
    │   ├── ImuDriftPlotView.java     (T3 — desenha trajetória 2D + erro×tempo no Canvas)
    │   └── DisparityClickView.java   (reusa lógica do DepthActivity do M4)
    └── export/
        └── PlyExporterMetric.java    (T7 — exporta nuvem .ply com Z métrico real)
```

---

## 2. Padrões de Design

| Padrão | Onde | Por quê |
|---|---|---|
| **MVVM** | toda Activity | Separação UI / lógica / dados |
| **State** | `VioState` enum | UI reflete claramente as fases (esperando keyframe, processando, etc.) |
| **Strategy** | `KeyframeCriteria` (BaselineParallax / Strict / Loose), `ZuptStrategy`, `OrbMatcherStrategy` | Trocar critérios sem mudar consumidor |
| **Singleton (Holder)** | `SyncedDataRepository`, `KeyframePairRepository`, `VioCsvLogger` | Estado global thread-safe, sobrevive a rotação |
| **Factory** | `KeyframeCriteriaFactory` | Presets STRICT/BALANCED/LOOSE |
| **Builder** | `KeyframeCriteria.Builder` (s_min, p_min, max_rot_deg, min_inliers) | Múltiplos parâmetros |
| **Template Method** | `VioPipeline` (no `VioProcessingActivity` ou em classe separada) | 5 etapas fixas: matching → E → recoverPose → escala IMU → retificar/SGBM |
| **Observer** (LiveData) | ViewModels | Reatividade UI |
| **Repository** | 3 repositórios | Abstrai estado mutável global |

---

## 3. Implementação Ordenada (Tarefas T1–T8)

### 🔵 T1 — Calibração intrínseca (REUSAR Módulo 3)

**Status:** ✅ Já feito no Módulo 3.

**Ação:**
```java
CalibrationResult cr = CalibrationJsonStore.load(ctx);
if (cr == null) { /* mostrar dialog "Execute o Módulo 3 primeiro" */ }
Mat K = cr.getCameraMatrix();
Mat distCoeffs = cr.getDistCoeffs();
double fx = cr.getFx();
```

`VioMainActivity.onCreate()` verifica se `calibration.json` existe. Se não, exibir botão "Ir para Calibração".

### 🔵 T2 — Captura sincronizada câmera + IMU

#### 2.1 `sensor/SensorLogger.java`

```java
public class SensorLogger implements SensorEventListener {
    public static class Sample {
        public final long tNs;
        public final float[] v;
        public final int type;
        public Sample(long tNs, float[] v, int type) {
            this.tNs = tNs; this.v = v.clone(); this.type = type;
        }
    }

    private final List<Sample> accel = Collections.synchronizedList(new ArrayList<>());
    private final List<Sample> gyro  = Collections.synchronizedList(new ArrayList<>());
    private final List<Sample> rot   = Collections.synchronizedList(new ArrayList<>());

    public interface Listener {
        void onAccel(Sample s);
        void onGyro(Sample s);
        void onRotation(Sample s);
    }
    private Listener listener;
    public void setListener(Listener l) { this.listener = l; }

    public void start(SensorManager sm) {
        register(sm, Sensor.TYPE_LINEAR_ACCELERATION);
        register(sm, Sensor.TYPE_GYROSCOPE);
        register(sm, Sensor.TYPE_ROTATION_VECTOR);
    }
    public void stop(SensorManager sm) { sm.unregisterListener(this); }

    private void register(SensorManager sm, int type) {
        Sensor s = sm.getDefaultSensor(type);
        if (s != null) sm.registerListener(this, s, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override public void onSensorChanged(SensorEvent e) {
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
    @Override public void onAccuracyChanged(Sensor s, int a) {}

    public List<Sample> getAccel() { return new ArrayList<>(accel); }
    public List<Sample> getGyro()  { return new ArrayList<>(gyro); }
    public List<Sample> getRotation() { return new ArrayList<>(rot); }
    public void clear() { accel.clear(); gyro.clear(); rot.clear(); }
}
```

#### 2.2 `VioCaptureActivity.java` — CameraX com ImageAnalysis

Layout `vio_activity_capture.xml`:
- `PreviewView` ocupando ~70% da tela
- Topo: TextView de status (IDLE / WAITING_KEYFRAME / KEYFRAME_READY)
- Topo: TextView de **offset estimado câmera↔IMU** (ms)
- Lado direito: `ReadinessBarView` (barra de prontidão)
- Bottom: botões `[▶ Iniciar]` `[⏸ Pausar]` `[🎯 Capturar par]` `[➡ Processar]` `[🧪 Teste de drift IMU]`

```java
ImageAnalysis analysis = new ImageAnalysis.Builder()
    .setTargetResolution(new Size(1280, 720))
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .build();
analysis.setAnalyzer(executor, image -> {
    long tFrameNs = image.getImageInfo().getTimestamp(); // base elapsedRealtimeNanos
    Mat rgb = yuvToRgbMat(image);
    SyncedDataRepository.getInstance().addFrame(new FrameSample(tFrameNs, rgb));
    imuNavigator.update(); // recalcula pose corrente
    triggerKeyframeAttempt(); // se aplicável
    image.close();
});
```

Implementar `yuvToRgbMat(ImageProxy)` em `vio/StereoUtils.java` — utilitário padrão de conversão YUV_420_888 → Mat RGB.

#### 2.3 Sincronização — validação empírica

Adicionar **uma função de diagnóstico** chamada uma vez na inicialização da Activity:

```java
public void diagnoseClockOffset() {
    long elapsedNs = SystemClock.elapsedRealtimeNanos();
    long sensorNs = lastSensorTimestamp;
    long offsetMs = (elapsedNs - sensorNs) / 1_000_000;
    if (Math.abs(offsetMs) > 50) {
        logger.log("CLOCK_OFFSET_WARNING", "Offset de " + offsetMs + " ms detectado");
    }
    logger.logDetailed("CLOCK_BASE_CHECK", null, null, ..., notes="offset_ms=" + offsetMs);
    runOnUiThread(() -> tvOffset.setText("Offset: " + offsetMs + " ms"));
}
```

**Critério de aceite T2:**
- [ ] CameraX faz preview + ImageAnalysis funcionando
- [ ] IMU registra LINEAR_ACCEL + GYRO + ROTATION_VECTOR em SENSOR_DELAY_FASTEST
- [ ] Frames têm timestamp em `elapsedRealtimeNanos`
- [ ] Offset câmera↔IMU exibido na tela (esperado: |offset| < 50 ms em phones modernos)
- [ ] CSV registra `IMU_STARTED`, `IMU_STOPPED`, `CLOCK_BASE_CHECK`, taxa de amostragem dos 3 sensores

### 🔵 T3 — Navegação IMU (ImuNavigator) + ZUPT

#### 3.1 `pipeline/ImuNavigator.java`

```java
public class ImuNavigator {
    private final double[] p = {0,0,0};
    private final double[] v = {0,0,0};
    private float[] q = {0,0,0,1};
    private long tPrev = -1;
    private double restAccum = 0;

    private static final double EPS = 0.08;       // m/s² — limiar de "em repouso"
    private static final double REST_T = 0.30;    // s — tempo mínimo de repouso pra disparar ZUPT
    private static final double DT_MAX = 0.1;     // s — descarta amostras com lacuna > 100ms

    private final List<Pose> history = Collections.synchronizedList(new ArrayList<>());
    private final ZuptStrategy zupt;

    public ImuNavigator(ZuptStrategy zupt) {
        this.zupt = zupt;
    }

    public synchronized void onAccel(long tNs, float[] aBody, float[] qNow) {
        q = qNow;
        if (tPrev < 0) { tPrev = tNs; return; }
        double dt = (tNs - tPrev) * 1e-9;
        tPrev = tNs;
        if (dt <= 0 || dt > DT_MAX) return;

        double[] aNav = rotateBodyToNav(aBody, q);
        double mag = Math.sqrt(aBody[0]*aBody[0] + aBody[1]*aBody[1] + aBody[2]*aBody[2]);

        if (zupt.isAtRest(mag, dt, restAccum)) {
            v[0] = v[1] = v[2] = 0;
            restAccum = 0;
        } else if (mag < EPS) {
            restAccum += dt;
        } else {
            restAccum = 0;
        }

        for (int i = 0; i < 3; i++) {
            p[i] += v[i] * dt + 0.5 * aNav[i] * dt * dt;
            v[i] += aNav[i] * dt;
        }
        history.add(new Pose(tNs, p.clone(), q.clone()));
    }

    public synchronized double distanceBetween(long tA, long tB) {
        double[] pa = posAt(tA), pb = posAt(tB);
        return Math.sqrt(sq(pb[0]-pa[0]) + sq(pb[1]-pa[1]) + sq(pb[2]-pa[2]));
    }

    public synchronized Pose getPoseAt(long tNs) {
        // interpolação linear entre as duas poses mais próximas
        // ...
    }

    public synchronized void resetWithPose(double[] pNew, float[] qNew) {
        System.arraycopy(pNew, 0, p, 0, 3);
        v[0] = v[1] = v[2] = 0;
        q = qNew.clone();
    }

    private double[] rotateBodyToNav(float[] a, float[] qn) {
        float[] R = new float[9];
        SensorManager.getRotationMatrixFromVector(R, qn);
        return new double[]{
            R[0]*a[0] + R[1]*a[1] + R[2]*a[2],
            R[3]*a[0] + R[4]*a[1] + R[5]*a[2],
            R[6]*a[0] + R[7]*a[1] + R[8]*a[2]
        };
    }
    private static double sq(double x) { return x*x; }
}
```

#### 3.2 `VioImuTestActivity.java` — Experimento OBRIGATÓRIO de drift

Activity dedicada para o **experimento de drift** exigido pela atividade:
- Botão "▶ Iniciar gravação" → começa a registrar IMU
- Usuário move o phone **1 m para frente e volta** (instrução na tela)
- Botão "⏹ Parar" → para gravação
- Exibe gráfico (em `ImuDriftPlotView`) de:
  - Trajetória 2D no plano XY
  - Erro de posição vs tempo (sem ZUPT vs com ZUPT — 2 curvas)
- Botão "💾 Salvar dados" → exporta CSV com poses
- Log do erro final no CSV

**Critério de aceite T3:**
- [ ] `ImuNavigator` integra LINEAR_ACCEL + ROTATION_VECTOR corretamente
- [ ] ZUPT detecta repouso (variância < EPS por > REST_T) e zera velocidade
- [ ] Gráfico de drift mostrado (esperado: ~10-30% de erro de posição em 5-10s sem ZUPT)
- [ ] CSV registra `IMU_DRIFT_TEST` com erro final em metros (com/sem ZUPT)

### 🔵 T4 — Seleção automática do par estéreo

#### 4.1 `strategy/KeyframeCriteria.java`

```java
public interface KeyframeCriteria {
    boolean isReady(double baseline_m, double parallax_px, int matches);
    double getMinBaseline();
    double getMinParallax();
    int getMinMatches();
}

public class BaselineParallaxStrategy implements KeyframeCriteria {
    public final double sMin, pMin;
    public final int matchMin;
    // ... getters + isReady = baseline >= sMin && parallax >= pMin && matches >= matchMin
}
```

#### 4.2 `factory/KeyframeCriteriaFactory.java`

```java
public class KeyframeCriteriaFactory {
    public static KeyframeCriteria strict()   { return new BaselineParallaxStrategy(0.10, 30, 100); }
    public static KeyframeCriteria balanced() { return new BaselineParallaxStrategy(0.05, 20,  80); }
    public static KeyframeCriteria loose()    { return new BaselineParallaxStrategy(0.03, 10,  50); }
}
```

#### 4.3 `pipeline/KeyframeSelector.java`

```java
public class KeyframeSelector {
    private final KeyframeCriteria criteria;
    private final OrbMatcherStrategy orb;
    private FrameSample ref;

    public KeyframePair tryPair(FrameSample cur, ImuNavigator imu, Mat K, Mat distCoeffs) {
        if (ref == null) { ref = cur; return null; }
        double baseline = imu.distanceBetween(ref.tNs, cur.tNs);
        if (baseline < criteria.getMinBaseline()) return null;

        MatchResult m = orb.match(ref.rgb, cur.rgb, K, distCoeffs);
        if (m.good.size() < criteria.getMinMatches()) return null;

        double parallax = medianPixelShift(m);
        if (parallax < criteria.getMinParallax()) return null;

        return new KeyframePair(ref, cur, baseline, parallax, m);
    }

    public void resetReference(FrameSample newRef) { this.ref = newRef; }
}
```

#### 4.4 `ui/ReadinessBarView.java`

View customizada que recebe `setReadiness(double baselineRatio, double parallaxRatio)` (cada um entre 0 e 1). Desenha duas barras horizontais (Canvas) com cores que viram verde quando passam de 1.0.

**Critério de aceite T4:**
- [ ] `KeyframeSelector` aceita par só quando baseline + parallax + matches atendem critério
- [ ] Barra de prontidão no app mostra evolução em tempo real
- [ ] CSV registra `KEYFRAME_REJECTED` (com motivo) e `KEYFRAME_ACCEPTED`
- [ ] 3 cenas com pelo menos 1 par aceito cada

### 🔵 T5 — Matching ORB (REUTILIZAR)

#### 5.1 `strategy/OrbMatcherStrategy.java`

Cópia da lógica de matching ORB+Lowe do `StereoPipeline` do Módulo 4, encapsulada em classe própria:

```java
public class OrbMatcherStrategy {
    public static class MatchResult {
        public MatOfKeyPoint kpL, kpR;
        public Mat undL, undR;
        public List<DMatch> good;
    }

    public MatchResult match(Mat imgL, Mat imgR, Mat K, Mat distCoeffs) {
        Mat undL = new Mat(), undR = new Mat();
        Calib3d.undistort(imgL, undL, K, distCoeffs);
        Calib3d.undistort(imgR, undR, K, distCoeffs);

        Mat grayL = new Mat(), grayR = new Mat();
        Imgproc.cvtColor(undL, grayL, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(undR, grayR, Imgproc.COLOR_BGR2GRAY);

        ORB orb = ORB.create(2000);
        MatOfKeyPoint kpL = new MatOfKeyPoint(), kpR = new MatOfKeyPoint();
        Mat desL = new Mat(), desR = new Mat();
        orb.detectAndCompute(grayL, new Mat(), kpL, desL);
        orb.detectAndCompute(grayR, new Mat(), kpR, desR);

        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        List<MatOfDMatch> knn = new ArrayList<>();
        matcher.knnMatch(desL, desR, knn, 2);

        List<DMatch> good = new ArrayList<>();
        for (MatOfDMatch m : knn) {
            DMatch[] arr = m.toArray();
            if (arr.length >= 2 && arr[0].distance < 0.75f * arr[1].distance) good.add(arr[0]);
        }
        MatchResult r = new MatchResult();
        r.kpL = kpL; r.kpR = kpR; r.undL = undL; r.undR = undR; r.good = good;
        grayL.release(); grayR.release(); desL.release(); desR.release();
        return r;
    }
}
```

**Critério de aceite T5:**
- [ ] Imagem com matches sobrepostos exibida na ProcessingActivity
- [ ] Contagem antes/depois do ratio test no CSV (`ORB_DONE`)

### 🔵 T6 — Pose relativa (E + recoverPose + escala IMU)

#### 6.1 `pipeline/RelativePoseEstimator.java`

```java
public class RelativePoseEstimator {

    public RelativePoseResult estimate(KeyframePair pair, Mat K, Mat distCoeffs,
                                        ImuNavigator imu) {
        OrbMatcherStrategy.MatchResult m = pair.matches;
        List<Point> ptsA = new ArrayList<>(), ptsB = new ArrayList<>();
        for (DMatch d : m.good) {
            ptsA.add(m.kpL.toList().get(d.queryIdx).pt);
            ptsB.add(m.kpR.toList().get(d.trainIdx).pt);
        }

        // 1. Undistort para coordenadas normalizadas
        MatOfPoint2f pA = new MatOfPoint2f(); pA.fromList(ptsA);
        MatOfPoint2f pB = new MatOfPoint2f(); pB.fromList(ptsB);
        MatOfPoint2f unA = new MatOfPoint2f(), unB = new MatOfPoint2f();
        Calib3d.undistortPoints(pA, unA, K, distCoeffs);
        Calib3d.undistortPoints(pB, unB, K, distCoeffs);

        // 2. Matriz Essencial com RANSAC
        Mat mask = new Mat();
        Mat E = Calib3d.findEssentialMat(unA, unB, K, Calib3d.RANSAC, 0.999, 1.0, mask);

        // 3. recoverPose → R_rel, t_rel (|t_rel|=1)
        Mat R_rel = new Mat(), t_rel = new Mat();
        int inliers = Calib3d.recoverPose(E, unA, unB, K, R_rel, t_rel, mask);

        // 4. Validar movimento não-degenerado
        double angleDeg = rotationAngleDeg(R_rel);
        if (angleDeg > 15.0) {
            // log + return result com flag de "degenerated rotation"
        }

        // 5. Escala métrica da IMU
        double s = imu.distanceBetween(pair.A.tNs, pair.B.tNs);
        Mat t_metric = new Mat();
        Core.multiply(t_rel, new Scalar(s), t_metric);

        return new RelativePoseResult(R_rel, t_metric, inliers, angleDeg, s);
    }

    private double rotationAngleDeg(Mat R) {
        // theta = arccos((tr(R) - 1) / 2) em graus
        double tr = R.get(0,0)[0] + R.get(1,1)[0] + R.get(2,2)[0];
        double cos = Math.max(-1, Math.min(1, (tr - 1) / 2));
        return Math.toDegrees(Math.acos(cos));
    }
}
```

**Critério de aceite T6:**
- [ ] E estimado com RANSAC; inliers > 50 (ideal) ou warning se < 50
- [ ] R_rel e t_metric retornados, com escala em metros real
- [ ] Ângulo de rotação relativa logado; par rejeitado se > 15°
- [ ] CSV registra `E_ESTIMATED`, `POSE_RECOVERED` (com inliers, angleDeg, scale_m)

### 🔵 T7 — Retificação CALIBRADA + Disparidade + Profundidade Q

#### 7.1 `pipeline/CalibratedRectifier.java`

```java
public class CalibratedRectifier {

    public CalibratedRectifyResult rectify(Mat imgL, Mat imgR, Mat K, Mat distCoeffs,
                                            Mat R_rel, Mat t_metric, Size imgSize) {
        Mat R1 = new Mat(), R2 = new Mat();
        Mat P1 = new Mat(), P2 = new Mat();
        Mat Q  = new Mat();
        Calib3d.stereoRectify(K, distCoeffs, K, distCoeffs, imgSize,
                R_rel, t_metric, R1, R2, P1, P2, Q,
                Calib3d.CALIB_ZERO_DISPARITY, 0);

        Mat m1x = new Mat(), m1y = new Mat();
        Mat m2x = new Mat(), m2y = new Mat();
        Calib3d.initUndistortRectifyMap(K, distCoeffs, R1, P1, imgSize, CvType.CV_32FC1, m1x, m1y);
        Calib3d.initUndistortRectifyMap(K, distCoeffs, R2, P2, imgSize, CvType.CV_32FC1, m2x, m2y);

        Mat rectL = new Mat(), rectR = new Mat();
        Imgproc.remap(imgL, rectL, m1x, m1y, Imgproc.INTER_LINEAR);
        Imgproc.remap(imgR, rectR, m2x, m2y, Imgproc.INTER_LINEAR);

        m1x.release(); m1y.release(); m2x.release(); m2y.release();
        return new CalibratedRectifyResult(rectL, rectR, R1, R2, P1, P2, Q);
    }
}
```

#### 7.2 `pipeline/DepthFromQ.java`

```java
public static Mat reproject(Mat disp32f, Mat Q) {
    Mat xyz = new Mat();
    Calib3d.reprojectImageTo3D(disp32f, xyz, Q, true);
    return xyz;
}
```

#### 7.3 SGBM — REUTILIZAR `estereo/strategy/SgbmDisparityStrategy.java`

Não duplicar. Importar diretamente:
```java
import com.example.visionproject.estereo.strategy.SgbmDisparityStrategy;
import com.example.visionproject.estereo.factory.SgbmParamsFactory;
```

#### 7.4 `export/PlyExporterMetric.java`

Como agora temos a matriz Q válida, podemos exportar Z em metros **reais** (não saturados). Reutilizar lógica do `PlyExporter` do M4, mas usando a saída de `reprojectImageTo3D` em vez do triângulo manual.

**Critério de aceite T7:**
- [ ] Par retificado com linhas horizontais sobrepostas exibido
- [ ] Mapa de disparidade colormap exibido
- [ ] Q salva e usada para reprojectImageTo3D
- [ ] Profundidade Z (em metros REAIS via Q) exibida ao clicar em pontos
- [ ] PNG dos artefatos salvos em `Pictures/VisionProject/vio/`

### 🔵 T8 — Fusão VIO + Análise crítica

#### 8.1 `pipeline/VioFusion.java`

```java
public class VioFusion {
    private static final float ALPHA = 0.05f;

    private double[] fusedP = {0,0,0};
    private float[] fusedQ = {0,0,0,1};

    public void fusePose(double[] poseVisual, float[] qVisual,
                          double[] poseIMU, float[] qIMU,
                          long tNs, ImuNavigator imu) {
        for (int i = 0; i < 3; i++) {
            fusedP[i] = ALPHA * poseVisual[i] + (1 - ALPHA) * poseIMU[i];
        }
        fusedQ = QuaternionUtil.slerp(qIMU, qVisual, ALPHA);
        // Corrige deriva: atualiza estado da IMU com a fusão
        imu.resetWithPose(fusedP, fusedQ);
    }

    public double[] getFusedPosition() { return fusedP; }
    public float[] getFusedOrientation() { return fusedQ; }
}
```

#### 8.2 Análise crítica obrigatória

Em `VioProcessingActivity`, ao final do processamento de cada par:
- Toca em ponto no mapa de disparidade
- Mostra Z_estimado (via Q reproject)
- Botão "📏 Comparar com fita métrica" → dialog → digita Z_real → app calcula erro relativo
- Loga `SCALE_ERROR_ANALYSIS` no CSV com Z_est, Z_real, error_pct

**Critério de aceite T8:**
- [ ] Fusão complementar implementada (α=0.05 default, ajustável)
- [ ] Dialog de comparação com fita métrica funcional
- [ ] CSV registra erro de escala por cena

---

## 4. Logger CSV (estrutura)

**Arquivo:** `Documents/VisionProject/vio_session_VIO_<yyyyMMdd_HHmmss>.csv`

**Header (estende os anteriores):**
```
session_id,timestamp_ms,timestamp_iso,event_type,
scene_id,frame_tns,sensor_type,
ax,ay,az,gx,gy,gz,qx,qy,qz,qw,
baseline_m,parallax_px,matches_good,inliers_ransac,rotation_deg,
fx_px,Z_est_m,Z_real_m,error_pct,
file_path,device_model,android_version,notes
```

**Eventos:**
- `SESSION_START`
- `CLOCK_BASE_CHECK` (offset detectado)
- `IMU_STARTED`, `IMU_STOPPED`
- `IMU_DRIFT_TEST` (erro_final_sem_zupt, erro_final_com_zupt)
- `KEYFRAME_REJECTED` (motivo: baseline / parallax / matches / rotation)
- `KEYFRAME_ACCEPTED` (baseline, parallax, matches)
- `ORB_DONE`, `E_ESTIMATED`, `POSE_RECOVERED`
- `RECTIFY_CALIBRATED_DONE`, `SGBM_RUN` (×3 presets)
- `DEPTH_QUERY` (Z_est em metros REAIS)
- `SCALE_ERROR_ANALYSIS` (Z_est vs Z_real, erro %)
- `VIO_FUSION_STEP` (peso α, posição fundida)
- `PLY_EXPORTED_METRIC`
- `SESSION_CLOSED`

---

## 5. Critérios de Aceite (DoD)

### Regressão (smoke test obrigatório)
- [ ] PCC, ModeloCamera, Calibração, Estéreo continuam funcionando
- [ ] CSVs anteriores continuam sendo gerados

### Funcional VIO
- [ ] Sincronização câmera↔IMU validada empiricamente (offset < 50 ms documentado)
- [ ] ImuNavigator integra corretamente, ZUPT funciona
- [ ] Experimento de drift IMU realizado (gráfico + CSV)
- [ ] Seletor de keyframe rejeita movimentos degenerados (frontal/rotação pura)
- [ ] E + recoverPose calculados com inliers > 50 em pelo menos 1 cena
- [ ] Retificação CALIBRADA aplicada (linhas horizontais alinhadas)
- [ ] SGBM 3 presets gerados
- [ ] Z em metros REAIS via Q (não saturado)
- [ ] Dialog "Comparar com fita" funciona
- [ ] PLY métrico exportado
- [ ] 3 cenas processadas

### Persistência
- [ ] CSV completo gerado em `Documents/VisionProject/`
- [ ] Artefatos PNG em `Pictures/VisionProject/vio/`
- [ ] PLY em `Pictures/VisionProject/`

---

## 6. Anti-padrões

- ❌ **Não** rodar SGBM/E/RANSAC/ORB na main thread
- ❌ **Não** misturar timestamps de relógios diferentes (sempre `elapsedRealtimeNanos`)
- ❌ **Não** assumir que o frame mais recente tem timestamp = agora (CameraX tem latência ~30ms)
- ❌ **Não** quebrar API pública de classes existentes (PCC/ModeloCamera/Calibração/Estéreo)
- ❌ **Não** usar `stereoRectifyUncalibrated` aqui — usar `stereoRectify` (calibrado)
- ❌ **Não** ignorar offset de relógio se > 50ms — corrigir explicitamente subtraindo
- ❌ **Não** chamar `recoverPose` sem verificar inliers (> 50)
- ❌ **Não** integrar IMU sem ZUPT — drift explode em segundos
- ❌ **Não** usar baseline da primeira tentativa de movimento — aguarda critérios
- ❌ **Não** importar `org.opencv.features2d.SIFT` (exige opencv-contrib não disponível)

---

## 7. Cheat-sheet de implementação

1. Criar pacote `vio/` com 8 subpacotes
2. **Modelos** primeiro: `ImuSample`, `Pose`, `FrameSample`, `KeyframePair`, `RelativePoseResult`, etc.
3. `sensor/SensorLogger` — base de tudo
4. `pipeline/ImuNavigator` + `strategy/ZuptStrategy` (T3)
5. `VioImuTestActivity` — testa ImuNavigator isoladamente
6. `pipeline/KeyframeSelector` + `strategy/KeyframeCriteria` + Factory (T4)
7. `strategy/OrbMatcherStrategy` (T5)
8. `pipeline/RelativePoseEstimator` (T6)
9. `pipeline/CalibratedRectifier` + `pipeline/DepthFromQ` (T7)
10. `pipeline/VioFusion` (T8)
11. `repository/SyncedDataRepository`, `KeyframePairRepository`, `VioCsvLogger`
12. `VioCaptureActivity` + `VioProcessingActivity` + `VioMainActivity`
13. Layouts XML + strings + manifest (adicionar 4 Activities)
14. MainActivity — adicionar botão "VIO (Estéreo sem suporte)"
15. Smoke test completo
16. **Antes de declarar pronto:** verificar **base de tempo comum** (passo crítico mais subestimado)

---

## 8. O que NÃO está no escopo

- ❌ Nova calibração (reusa Módulo 3)
- ❌ SIFT/SURF (sem opencv-contrib)
- ❌ Filtro de Kalman estendido (somente complementar — α fixo)
- ❌ Stereo calibration (`stereoCalibrate`) — usa pose relativa estimada de E
- ❌ Loop closure / global bundle adjustment (fora do escopo, é MSCKF/VINS)
- ❌ Bias online do giroscópio (assume que ROTATION_VECTOR já compensa)
- ❌ Relatório PDF (eu monto depois com seus dados)
- ❌ Apresentação .pptx (eu monto depois)
- ❌ Vídeo YouTube (você grava)

---

**Autor da especificação:** Vinicius L. Alvarenga
**Versão:** 1.0
**Data:** Junho/2026
**Pré-requisitos:** Módulos 1, 2, 3 e 4 já implementados e funcionais.
