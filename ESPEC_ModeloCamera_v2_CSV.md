# ESPECIFICAÇÃO TÉCNICA v2 — Logging CSV no `ModeloCameraActivity`
**Projeto:** VisionProject (Android, Java + OpenCV4Android)
**Módulo:** 2 — Formação da Imagem e Modelo de Câmera
**Pré-requisito:** ESPEC v1 (`ESPEC_ModeloCamera.md`) já implementada — **NÃO PODE SER REGREDIDA**

---

## 0. Princípios Não-Negociáveis (revisão)

1. **Zero alteração no PccModule** (Módulo 1) — continua valendo.
2. **Zero alteração nas classes existentes** do `modelocamera/` que já funcionam — apenas **adicionar** novas classes e **expandir** o `ModeloCameraViewModel` com métodos novos. Nunca remover ou renomear método público existente.
3. **Reutilizar o padrão de CSV do PCC.** O Módulo 1 já grava CSVs no padrão `Pictures/VisionProject/` (ou diretório análogo). O novo logger deve gravar **na mesma pasta** para manter consistência de UX. Se o PCC grava em outro lugar, **adotar o mesmo lugar**.
4. **Padrões de design:**
   - **Observer/Listener** para desacoplar o logger das fontes de evento (ViewModel publica, Logger escuta).
   - **Singleton** (Holder pattern thread-safe) para o Logger — uma instância única por sessão do app.
   - **Strategy** (opcional, mas recomendado) para o formato de saída — caso queira no futuro suportar JSON além de CSV.
5. **I/O sempre fora do main thread.** Todas as escritas de arquivo via `ExecutorService` (single-thread executor).
6. **Compatibilidade de armazenamento.** Usar `MediaStore` para Android Q+ (API 29+) e `Environment.getExternalStoragePublicDirectory()` apenas como fallback. **Nunca** caminho absoluto hardcoded.

---

## 1. Estrutura de Pacotes (somente adições)

```
com.example.visionproject/
├── modelocamera/                         [JÁ EXISTE — apenas adicionar abaixo]
│   ├── ModeloCameraActivity.java         [JÁ EXISTE — adicionar 1 chamada nova]
│   ├── ModeloCameraViewModel.java        [JÁ EXISTE — adicionar listener wiring]
│   └── logger/                           [NOVO]
│       ├── CalibrationSessionLogger.java (Singleton + Observer)
│       ├── SessionEvent.java             (sealed/abstract)
│       └── event/
│           ├── CalibrationLoadedEvent.java
│           ├── ImageCapturedEvent.java
│           ├── ImageUndistortedEvent.java
│           ├── PointUndistortedEvent.java
│           ├── EpipolarPointAddedEvent.java
│           └── SessionClosedEvent.java
└── shared/                               [JÁ EXISTE — adicionar abaixo]
    ├── csv/                              [NOVO]
    │   ├── CsvWriter.java                (utilitário thread-safe)
    │   └── CsvFormatter.java             (formatação numérica padronizada)
    └── DeviceInfoProvider.java           (NOVO — modelo do device, versão Android)
```

---

## 2. Padrões de Design Aplicados nesta v2

| Padrão | Onde | Por quê |
|---|---|---|
| **Singleton** (Holder) | `CalibrationSessionLogger` | Uma única sessão de log por execução do app, garantia de thread-safety |
| **Observer** (Listener) | `CalibrationSessionLogger` registra-se em ViewModel | Desacoplamento: ViewModel não conhece Logger, apenas publica eventos |
| **Strategy** (opcional) | `CsvFormatter` vs futuro `JsonFormatter` | Suportar formatos alternativos sem mudar Logger |
| **Sealed class hierarchy** (via abstract) | `SessionEvent` + subclasses | Cada tipo de evento tem campos específicos, polimorfismo limpo |
| **Immutable events** | Todas as `*Event` | Eventos são fatos, nunca mutáveis depois de criados |

---

## 3. Modelo de Dados — `SessionEvent` e subtipos

### 3.1 `logger/SessionEvent.java` (abstract)

```java
public abstract class SessionEvent {
    public final long timestampMs;        // System.currentTimeMillis()
    public final String eventType;        // "IMAGE_CAPTURED", etc.

    protected SessionEvent(String eventType) {
        this.timestampMs = System.currentTimeMillis();
        this.eventType = eventType;
    }

    /** Cada evento sabe se imprimir como linha CSV. */
    public abstract String toCsvRow(CsvFormatter fmt);
}
```

### 3.2 Subtipos concretos (todos imutáveis, todos com `toCsvRow()`)

| Classe | Campos próprios | eventType |
|---|---|---|
| `CalibrationLoadedEvent` | `CameraIntrinsics K`, `DistortionCoefficients D` | `"CALIBRATION_LOADED"` |
| `ImageCapturedEvent` | `String filename`, `int width`, `int height`, `String source` ("CAMERA"\|"GALLERY") | `"IMAGE_CAPTURED"` |
| `ImageUndistortedEvent` | `String origFilename`, `String corrFilename`, `int width`, `int height`, `long elapsedMs` | `"IMAGE_UNDISTORTED"` |
| `PointUndistortedEvent` | `Point original`, `Point corrected`, `double distFromCenter` | `"POINT_UNDISTORT"` |
| `EpipolarPointAddedEvent` | `Point coords`, `int index` (0–4), `int colorRgb`, `String label` | `"EPIPOLAR_POINT"` |
| `SessionClosedEvent` | `int totalEvents`, `String reason` ("USER"\|"DESTROY") | `"SESSION_CLOSED"` |

**Importante:** todos os campos `final`. Sem setters.

---

## 4. Especificação do CSV de saída

### 4.1 Localização
```
<diretório-PCC>/mc_session_<yyyyMMdd_HHmmss>.csv
```
Exemplo: `Pictures/VisionProject/mc_session_20260509_165130.csv`

**Mesmo diretório** que os CSVs do PCC. Se o PCC usa `Environment.getExternalStorageDirectory()/VisionProject/`, usar o mesmo. Se usa MediaStore com `RELATIVE_PATH=Pictures/VisionProject/`, usar o mesmo.

### 4.2 Estrutura do arquivo

Linha de cabeçalho única, seguida de N linhas de eventos. **Sem bloco comentado** — apenas CSV puro (Excel-friendly):

```csv
session_id,timestamp_ms,timestamp_iso,event_type,fx,fy,cx,cy,k1,k2,p1,p2,k3,res_w,res_h,fov_h_deg,distortion_type,x_orig,y_orig,x_corr,y_corr,delta_x,delta_y,dist_from_center,color_hex,point_index,image_filename,image_source,elapsed_ms,device_model,android_version,notes
mc_20260509_165130,1715281890000,2026-05-09T16:51:30,SESSION_START,,,,,,,,,,,,,,,,,,,,,,,,,Samsung SM-G975F,14,App started
mc_20260509_165130,1715281890123,2026-05-09T16:51:30,CALIBRATION_LOADED,1408.5,1410.2,955.8,538.1,0.12,-0.08,0.001,0.0005,0.0,1912,1076,68.6,PINCUSHION,,,,,,,,,,,,,Samsung SM-G975F,14,Defaults from IntrinsicsFactory
mc_20260509_165130,1715281902000,2026-05-09T16:51:42,IMAGE_CAPTURED,,,,,,,,,,2592,1944,,,,,,,,,,,,grid_a4_orig_001.png,CAMERA,,Samsung SM-G975F,14,
mc_20260509_165130,1715281902350,2026-05-09T16:51:42,IMAGE_UNDISTORTED,,,,,,,,,,2592,1944,,,,,,,,,,,,grid_a4_corr_001.png,,87,Samsung SM-G975F,14,
mc_20260509_165130,1715281910500,2026-05-09T16:51:50,POINT_UNDISTORT,,,,,,,,,,,,,,1200,800,1198.06,797.87,-1.94,-2.13,316.4,,,,,,Samsung SM-G975F,14,
mc_20260509_165130,1715281915123,2026-05-09T16:51:55,EPIPOLAR_POINT,,,,,,,,,,,,,,640,540,,,,,,#FF0000,0,,,,Samsung SM-G975F,14,first point
mc_20260509_165130,1715281917456,2026-05-09T16:51:57,EPIPOLAR_POINT,,,,,,,,,,,,,,800,400,,,,,,#FFA500,1,,,,Samsung SM-G975F,14,
mc_20260509_165130,1715281920789,2026-05-09T16:52:00,EPIPOLAR_POINT,,,,,,,,,,,,,,1100,600,,,,,,#FFFF00,2,,,,Samsung SM-G975F,14,
mc_20260509_165130,1715281923012,2026-05-09T16:52:03,EPIPOLAR_POINT,,,,,,,,,,,,,,1300,800,,,,,,#00FF00,3,,,,Samsung SM-G975F,14,
mc_20260509_165130,1715281925234,2026-05-09T16:52:05,EPIPOLAR_POINT,,,,,,,,,,,,,,1500,1000,,,,,,#0000FF,4,,,,Samsung SM-G975F,14,last point
mc_20260509_165130,1715281930000,2026-05-09T16:52:10,SESSION_CLOSED,,,,,,,,,,,,,,,,,,,,,,,,,Samsung SM-G975F,14,Total events: 9
```

### 4.3 Convenções de formatação

- **Decimal:** ponto (`1408.5`), nunca vírgula. Usar `Locale.US` no `String.format()`.
- **Doubles:** sempre 4 casas decimais (`%.4f`), exceto coordenadas de pixel (1 casa).
- **Coordenadas (x, y):** floats com 2 casas (`1198.06`).
- **Cores:** hex com `#` (`#FF0000`).
- **Strings com vírgula:** entre aspas duplas, escapar aspas internas com aspas duplas duplicadas (RFC 4180).
- **Campos vazios:** simplesmente nada entre as vírgulas (`,,`).
- **Encoding:** UTF-8 sem BOM.
- **Line ending:** `\r\n` (CRLF — compatibilidade Excel/Sheets).

---

## 5. Implementação ordenada (passo a passo)

### 🔹 Passo 1 — Utilitários compartilhados

#### 1.1 `shared/csv/CsvFormatter.java`
- Classe utilitária com métodos estáticos:
  - `static String fmtDouble(double v, int decimals)` — usa `Locale.US`
  - `static String fmtPoint(Point p)` — `"x.xx"` (apenas valor numérico, separado em duas chamadas)
  - `static String escape(String field)` — RFC 4180 (aspas se contém `,`, `"` ou `\n`)
  - `static String fmtTimestampIso(long ms)` — ISO-8601 local (`yyyy-MM-dd'T'HH:mm:ss`)
  - `static String fmtColor(int rgb)` — `"#RRGGBB"`

#### 1.2 `shared/csv/CsvWriter.java`
- Wrapper sobre `BufferedWriter` thread-safe.
- Construtor recebe `OutputStream` (do MediaStore) ou `File`.
- Método `synchronized void writeHeader(String[] columns)` — escreve a primeira linha com `\r\n`.
- Método `synchronized void writeRow(Map<String,String> row)` — escreve as colunas na ordem do header. Campos ausentes do mapa viram `""`.
- Método `synchronized void close()` — flush + close, idempotente.
- Internamente: `BufferedWriter` com buffer de 8 KB.

#### 1.3 `shared/DeviceInfoProvider.java`
- Métodos estáticos:
  - `static String getDeviceModel()` → `Build.MANUFACTURER + " " + Build.MODEL`
  - `static String getAndroidVersion()` → `Build.VERSION.RELEASE`
  - `static String getAppVersion(Context ctx)` → via `PackageInfo`
- Cache em `static final` para evitar recomputo.

### 🔹 Passo 2 — Eventos

#### 2.1 `logger/SessionEvent.java` (abstract — base)

#### 2.2 `logger/event/*.java`
- Cada classe é imutável (campos `final`).
- Cada uma implementa `toCsvRow(CsvFormatter fmt)` retornando uma `Map<String,String>` com **apenas as colunas que ela preenche**. O `CsvWriter` cuida do resto.
- Exemplo:
  ```java
  public class PointUndistortedEvent extends SessionEvent {
      public final Point original, corrected;
      public final double distFromCenter;

      public PointUndistortedEvent(Point original, Point corrected, double distFromCenter) {
          super("POINT_UNDISTORT");
          this.original = original;
          this.corrected = corrected;
          this.distFromCenter = distFromCenter;
      }

      @Override public Map<String,String> toCsvFields(CsvFormatter fmt) {
          Map<String,String> m = new LinkedHashMap<>();
          m.put("x_orig", fmt.fmtDouble(original.x, 1));
          m.put("y_orig", fmt.fmtDouble(original.y, 1));
          m.put("x_corr", fmt.fmtDouble(corrected.x, 2));
          m.put("y_corr", fmt.fmtDouble(corrected.y, 2));
          m.put("delta_x", fmt.fmtDouble(corrected.x - original.x, 2));
          m.put("delta_y", fmt.fmtDouble(corrected.y - original.y, 2));
          m.put("dist_from_center", fmt.fmtDouble(distFromCenter, 1));
          return m;
      }
  }
  ```

### 🔹 Passo 3 — Logger Singleton

#### 3.1 `logger/CalibrationSessionLogger.java`

Estrutura:
```java
public class CalibrationSessionLogger {

    private static class Holder {
        static final CalibrationSessionLogger INSTANCE = new CalibrationSessionLogger();
    }
    public static CalibrationSessionLogger getInstance() { return Holder.INSTANCE; }
    private CalibrationSessionLogger() {}

    private CsvWriter writer;
    private String sessionId;
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final AtomicInteger eventCount = new AtomicInteger(0);
    private boolean active = false;

    /** Inicia uma sessão; cria o CSV e escreve header + SESSION_START. */
    public synchronized void startSession(Context ctx) { /* ... */ }

    /** Recebe qualquer SessionEvent e enfileira escrita. */
    public synchronized void log(SessionEvent event, Context ctx) { /* ... */ }

    /** Encerra a sessão; escreve SESSION_CLOSED, fecha writer. */
    public synchronized void endSession(Context ctx, String reason) { /* ... */ }

    public boolean isActive() { return active; }
}
```

**Header do CSV (ordem fixa):**
```java
private static final String[] CSV_COLUMNS = {
    "session_id", "timestamp_ms", "timestamp_iso", "event_type",
    "fx","fy","cx","cy","k1","k2","p1","p2","k3",
    "res_w","res_h","fov_h_deg","distortion_type",
    "x_orig","y_orig","x_corr","y_corr","delta_x","delta_y","dist_from_center",
    "color_hex","point_index",
    "image_filename","image_source",
    "elapsed_ms",
    "device_model","android_version","notes"
};
```

**Lógica do `log(...)`:**
1. Se `!active`, ignorar (warn no Logcat).
2. Criar map base com `session_id`, `timestamp_ms`, `timestamp_iso`, `event_type`, `device_model`, `android_version`.
3. Mesclar com `event.toCsvFields(formatter)`.
4. Submeter ao `io.execute(() -> writer.writeRow(map))`.
5. Incrementar `eventCount`.

**Lógica do `startSession(...)`:**
1. Se já ativo, fazer `endSession(ctx, "RESTARTED")` antes.
2. Gerar `sessionId = "mc_" + timestamp formatado`.
3. Criar `OutputStream` via MediaStore (Android Q+) com `MediaStore.MediaColumns.RELATIVE_PATH = "Pictures/VisionProject"` e MIME `text/csv`.
4. Instanciar `writer = new CsvWriter(outputStream)`.
5. Escrever header.
6. Logar `CalibrationLoadedEvent` (obtido do `CalibrationRepository`) e um marker `SESSION_START`.
7. `active = true`.

**Lógica do `endSession(...)`:**
1. Se `!active`, retornar.
2. Logar `SessionClosedEvent(eventCount.get(), reason)`.
3. `io.submit(() -> writer.close())` + `io.shutdown()` + nova instância de executor.
4. `active = false`.

### 🔹 Passo 4 — Integração com ViewModel (mínima)

No `ModeloCameraViewModel.java` **já existente**, adicionar:

#### 4.1 Inicialização
No construtor (ou em `init()`), chamar:
```java
CalibrationSessionLogger.getInstance().startSession(applicationContext);
CalibrationSessionLogger.getInstance().log(
    new CalibrationLoadedEvent(repo.getIntrinsics(), repo.getDistortion()),
    applicationContext
);
```

> ⚠️ **NÃO** acoplar ViewModel a Activity diretamente. Passar `Application` (ou `Context.getApplicationContext()`) via `AndroidViewModel` ou via Factory. Se isso exigir mudar a herança do ViewModel, **ok mudar**, mas só essa classe.

#### 4.2 Em cada método existente, adicionar 1 linha de log

| Método existente do ViewModel | Adicionar |
|---|---|
| `onImageCaptured(Mat orig)` | Após salvar Bitmap original: `logger.log(new ImageCapturedEvent(filename, w, h, "CAMERA"), ctx)` |
| `onImageCaptured(Mat orig)` (após undistort) | `logger.log(new ImageUndistortedEvent(origFile, corrFile, w, h, elapsedMs), ctx)` |
| `onPointSelected(Point p)` | Após calcular ponto corrigido: `logger.log(new PointUndistortedEvent(p, pCorr, distCenter), ctx)` |
| `onEpipolarPointAdded(Point p, int idx, int color)` | `logger.log(new EpipolarPointAddedEvent(p, idx, color, label), ctx)` |
| `resetEpipolar()` | (sem log — apenas operação interna) |

> Os métodos do ViewModel **continuam tendo a mesma assinatura pública** — não quebrar contratos.

#### 4.3 Encerramento

No `ModeloCameraViewModel.onCleared()`:
```java
@Override protected void onCleared() {
    super.onCleared();
    CalibrationSessionLogger.getInstance().endSession(getApplication(), "DESTROY");
}
```

### 🔹 Passo 5 — UI: feedback ao usuário (opcional mas recomendado)

Na `ModeloCameraActivity`:
1. Adicionar botão `"📊 Salvar dados (CSV)"` no layout (`mc_activity_modelo_camera.xml`).
2. Click handler chama:
   ```java
   CalibrationSessionLogger.getInstance().endSession(getApplicationContext(), "USER");
   Toast.makeText(this, "CSV salvo em Pictures/VisionProject/", Toast.LENGTH_LONG).show();
   ```
3. Botão `"📤 Compartilhar CSV"` opcional, usando `Intent.ACTION_SEND` com URI do MediaStore.

> **Importante:** após `endSession("USER")`, o usuário **não pode** continuar logando eventos. Se precisar continuar, oferecer botão `"➕ Nova sessão"` que chama `startSession()` de novo.

### 🔹 Passo 6 — Testes unitários mínimos

Em `app/src/test/java/com/example/visionproject/modelocamera/logger/`:

#### 6.1 `PointUndistortedEventTest.java`
- Verifica que `toCsvFields()` produz mapa com 7 entradas.
- Verifica formatação numérica em Locale.US.

#### 6.2 `CsvFormatterTest.java`
- `fmtDouble(1408.5, 4)` → `"1408.5000"` (Locale.US, ponto decimal).
- `escape("foo,bar")` → `"\"foo,bar\""`.
- `escape("ok")` → `"ok"` (sem aspas).

#### 6.3 `CsvWriterTest.java`
- Cria writer em `ByteArrayOutputStream`, escreve header + 2 rows, valida output esperado.
- Verifica que `close()` é idempotente.

> Logger Singleton **não** precisa de teste unitário (depende de Context). Teste manual via app.

---

## 6. Critérios de Aceite (DoD)

- [ ] App compila sem warnings novos
- [ ] **PCC continua funcionando exatamente como antes** (smoke test obrigatório)
- [ ] **Tudo do ESPEC v1 continua funcionando** (capturar A4, undistort, comparação, epipolares)
- [ ] Ao abrir `ModeloCameraActivity`, é criado um arquivo `mc_session_<timestamp>.csv` em `Pictures/VisionProject/`
- [ ] Cabeçalho do CSV contém todas as colunas listadas na seção 4.3
- [ ] CSV recebe linha `CALIBRATION_LOADED` com fx, fy, cx, cy, k1..k3, distortion_type
- [ ] Capturar imagem → CSV ganha `IMAGE_CAPTURED` + `IMAGE_UNDISTORTED`
- [ ] Selecionar ponto → CSV ganha `POINT_UNDISTORT` com x_orig, y_orig, x_corr, y_corr, deltas, dist_from_center
- [ ] Adicionar 5 epipolares → CSV ganha 5 `EPIPOLAR_POINT` com index 0..4 e cor hex
- [ ] Botão "Salvar dados (CSV)" → encerra sessão com `SESSION_CLOSED` e mostra Toast
- [ ] CSV abre corretamente no Excel/Google Sheets (separador `,`, decimal `.`, encoding UTF-8)
- [ ] Rotação de tela **não** corrompe o CSV (Singleton sobrevive ao recreate da Activity)
- [ ] Saída do app via back button chama `endSession("DESTROY")` no `onCleared()`
- [ ] Testes unitários (`CsvFormatterTest`, `CsvWriterTest`, `PointUndistortedEventTest`) passam

---

## 7. Anti-padrões a evitar

- ❌ **Não** escrever no CSV no thread principal (sempre via `ExecutorService`).
- ❌ **Não** abrir/fechar arquivo a cada evento (manter writer aberto durante a sessão).
- ❌ **Não** compartilhar `BufferedWriter` entre threads sem `synchronized`.
- ❌ **Não** usar `Locale` default no `String.format` — sempre `Locale.US`.
- ❌ **Não** colocar lógica de CSV dentro de `SessionEvent` (somente formatação de campos próprios; orquestração no Logger).
- ❌ **Não** fazer o CSV depender de classes do Android (Bitmap, Uri, etc.) — só primitivas e POJOs.
- ❌ **Não** quebrar a API pública do `ModeloCameraViewModel` — apenas adicionar chamadas internas.
- ❌ **Não** criar segundo Singleton em paralelo (CalibrationRepository já é singleton; reaproveitar para obter K/D).

---

## 8. Resumo executivo (cheat-sheet)

1. Cria pacotes `modelocamera/logger/` + `modelocamera/logger/event/` + `shared/csv/`.
2. Implementa `CsvFormatter`, `CsvWriter`, `DeviceInfoProvider`.
3. Implementa `SessionEvent` abstract + 6 eventos concretos (todos imutáveis).
4. Implementa `CalibrationSessionLogger` (Singleton + ExecutorService + writer).
5. Adapta `ModeloCameraViewModel` para passar a herdar de `AndroidViewModel` (se ainda não), inicializa logger no construtor, adiciona 1 linha de `log(...)` em cada método existente (5 métodos).
6. Adiciona botão "Salvar dados (CSV)" no layout + handler na Activity.
7. Override `onCleared()` no ViewModel para chamar `endSession("DESTROY")`.
8. Roda testes unitários + smoke test manual.
9. Verifica todos os checkboxes da seção 6.

---

**Autor:** Vinicius L. Alvarenga
**Versão:** 2.0 (extensão CSV de v1)
**Data:** Maio/2026
