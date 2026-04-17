# Präsentations-Sprechernotizen — Active Learning for Vehicle Image Classification

> **Ziel:** 20–25 Minuten, Fokus auf *was wurde gebaut, warum so, wie funktioniert es*.
> Die wissenschaftlichen Grundlagen kennt der Betreuer — hier geht es um die konkreten Implementierungsentscheidungen und deren Begründungen.
> **Teil 6 (Beispiel-Experimente) bleibt leer** — wird nach Abschluss der Experimentläufe ergänzt.

---

## Folie 1 — Motivation & Problemstellung (~3 Min.)

### Das Kernproblem: Annotierung ist der Engpass

In der Computer Vision ist nicht das Training der Engpass — es ist das Labeling. Bei feingranularer Fahrzeugklassifikation (nicht nur „Auto vs. LKW", sondern Unterscheidung zwischen BMW 3er Coupé 2012 und BMW 3er Limousine 2012) braucht man Domänenexperten, die das manuell machen. Das Stanford Cars Dataset hat 196 solcher Klassen. 16.185 Bilder, jedes manuell klassifiziert. In der Praxis — bei einem Versicherer, einer Werkstatt-KI, einem Parkraum-Managementsystem — hat man typischerweise Hunderttausende Bilder, aber kein Budget, alle zu labeln.

### Warum Active Learning?

Die Idee: Statt alle Bilder zufällig labeln zu lassen, **fragt das Modell gezielt nach den Bildern, bei denen es am meisten lernen kann.** Das sind die Bilder nahe an Entscheidungsgrenzen — wo das Modell zwischen zwei Klassen schwankt. Jedes solche Label eliminiert mehr Hypothesen als ein „einfaches" Bild, das das Modell bereits richtig klassifizieren würde.

Die Forschungsfrage lautet: **Bringt diese gezielte Auswahl messbar mehr als zufälliges Sampling?** Und wenn ja — welche Unsicherheitsmaße (Entropy, Least Confidence, Margin) sind am effektivsten bei einem Problem mit 196 Klassen?

### Warum ein interaktives Tool?

Zweitens: Active Learning ist ein *iterativer* Prozess — Trainieren, Evaluieren, Abfragen, Annotieren, Wiederholen. Das passiert nicht einmal, sondern über 10+ Zyklen. Die meisten AL-Frameworks sind Skripte: man startet sie, wartet Stunden, und schaut am Ende auf eine Tabelle. Mein Ansatz: **Ein Dashboard, das den gesamten Zyklus transparent macht.** Man sieht live, wie das Modell trainiert, welche Bilder es auswählt, wie die Embedding-Landschaft sich verändert. Das ist nicht nur ein nettes Feature — es macht AL als Prozess *verständlich* und *debuggbar*.

### Was man hier zeigen kann (Überleitung zur Demo)

- Echtzeit-Trainingskurven während des Trainings
- Die vom Modell selektierten Bilder mit Unsicherheitswerten
- Pool-Statistiken: Wie verteilen sich gelabelte vs. ungelabelte Bilder über die Klassen?
- Nach dem Experiment: Accuracy-Kurven über alle Zyklen, UMAP-Projektionen, Konfusionsmatrizen

---

## Folie 2 — System-Überblick & Live-Demo (~3 Min.)

### Was das Tool kann

Das System ist ein vollständiges, interaktives Active-Learning-Framework mit vier Hauptbereichen:

**Sidebar (Konfiguration):**
Hier konfiguriert man alles vor dem Start. Modellarchitektur (aus einer kuratierten Liste von TIMM-Modellen — ResNet-Familie, EfficientNet, MobileNet), Sampling-Strategie (Entropy, Margin, Least Confidence, Random), Trainings-Hyperparameter (Epochen, Lernrate, Batch Size, Optimizer, Scheduler, Label Smoothing, Gradient Clipping) und AL-Einstellungen (Zyklenanzahl, Query-Batch-Größe, Initialer Pool, Reset-Modus). Das ist alles über die Sidebar konfigurierbar, ohne eine einzige YAML-Datei anzufassen.

**Main-Tab (Live-Training):**
Während das Experiment läuft, zeigt der Main-Tab Echtzeit-Updates: aktueller Zyklus und Epoche, Train/Val-Loss und Accuracy als Live-Charts, Lernraten-Verlauf, und Pool-Status (wie viele Bilder labeled/unlabeled). Das Polling-System aktualisiert diesen Tab je nach Zustand — schnell (0.5s) bei Querying/Annotation, langsam (1.5s) beim Training, gar nicht im Idle-Zustand.

**Results-Tab:**
Nach Abschluss eines Experiments zeigt dieser Tab die gespeicherten Ergebnisse von der Festplatte. Accuracy-Kurven über die Zyklen, Konfusionsmatrizen, Metriken pro Zyklus. Wichtig: Der Results-Tab scannt das `experiments/`-Verzeichnis — man kann also alte Runs jederzeit wieder laden und vergleichen.

**Compare-Tab:**
Hier kann man mehrere Runs nebeneinander vergleichen — z.B. Entropy vs. Random, oder ResNet-18 vs. ResNet-50. Genau das, was man für die Thesis-Figures braucht.

**Explorer-Tab:**
Zeigt die Klassenverteilung im gelabelten und ungelabelten Pool. Wichtig, um zu sehen, ob bestimmte Strategien bestimmte Klassen bevorzugen.

### Was man bei der Demo zeigen sollte

1. Sidebar öffnen, Konfiguration durchklicken — zeigt die Flexibilität
2. Quick-Test starten (4-Klassen-Config, 3 Zyklen, MobileNetV3 — läuft in Minuten)
3. Live-Trainingskurven zeigen, Pool-Statistiken
4. Nach Abschluss: Results-Tab öffnen, Accuracy-Kurve über Zyklen zeigen
5. Falls Zeit: Explorer-Tab, Klassenverteilung zeigen

### Zwei Betriebsmodi

Das System unterstützt zwei Modi:

- **Auto-Annotate (Simulationsmodus):** Ground-Truth-Labels werden automatisch zugewiesen. Das ist der Standardmodus für Experimente — kein manuelles Labeln nötig, der Loop läuft komplett durch.
- **Manual-Annotate (Human-in-the-Loop):** Der Loop pausiert nach dem Querying, zeigt die selektierten Bilder in einer Galerie, und wartet auf manuelle Annotation. Zeigt das AL-Konzept als interaktives Lehrtool.

`auto_annotate: false` stoppt den Loop tatsächlich und wartet — das war anfangs nicht offensichtlich und hat den Loop blockiert, bis die Annotation manuell durchgeführt wurde.

---

## Folie 3 — Active Learning Pipeline (~4 Min.)

### Der Zyklus im Detail

Der Active-Learning-Zyklus in meinem System besteht aus sechs Schritten, die in `active_loop.py` orchestriert werden:

**1. Initialisierung (`prepare_cycle`):**
- Beim allerersten Zyklus wird der initiale Pool aufgebaut. Dabei verwende ich **stratifizierte Initialisierung** (`stratified_init=True` in `data_manager.py`): Es wird garantiert, dass mindestens ein Bild pro Klasse im initialen gelabelten Pool ist. Bei 196 Klassen und einem initialen Pool von z.B. 500 Bildern bedeutet das: erst 196 Bilder (eins pro Klasse), dann 304 zufällig dazu. Das verhindert das „Cold Start"-Problem — wenn eine Klasse komplett fehlt, kann das Modell sie nie lernen und der Pool wird nie danach fragen.
- Der `DataManager` arbeitet rein indexbasiert — keine Datenduplikation. `_labeled_list` und `_unlabeled_list` sind einfach Listen von Integer-Indizes in das ImageFolder-Dataset. `PoolSubset` delegiert `__getitem__` an das Original-Dataset. Zero-Copy.

**2. Training (`train_single_epoch` × N):**
- Jede Epoche wird einzeln vom Worker-Thread über `al_loop.train_single_epoch(epoch)` getriggert. Nach jeder Epoche emittiert der Worker ein `EPOCH_DONE`-Event mit den Metriken, das die UI aktualisiert.
- Die ersten `freeze_backbone_epochs` Epochen (Default: 2) trainieren nur den Klassifikationskopf — der Backbone ist eingefroren. Danach wird der Backbone aufgetaut und der Optimizer **komplett neu erstellt** (nicht nur `requires_grad` umschalten). Grund: Die Backbone-Parameter hatten während des Einfrierens keinen Optimizer-State (keine Adam-Momente). Würde man den gleichen Optimizer behalten, hätten die Backbone-Parameter sofort große Updates wegen fehlender Momentum-Schätzung.
- Early Stopping wird nach jeder Epoche geprüft (Patience-Zähler in `trainer.py`), aber die Entscheidung wird im `active_loop` bzw. `worker` getroffen — nicht innerhalb des Trainers.

**3. Evaluation (`run_evaluation`):**
- Nach dem Training wird das beste Modell (nach Val-Accuracy) wiederhergestellt (`restore_best_model`) und auf dem festen Testset evaluiert.
- Metriken: Accuracy, Precision, Recall, F1 (alles weighted), ECE (Expected Calibration Error), Konfusionsmatrix.
- ECE misst, ob ein Modell, das 80% Confidence für eine Klasse zeigt, auch tatsächlich in 80% der Fälle richtig liegt. Wird mit 15 Bins berechnet.

**4. Finalize & UMAP (`finalize_cycle`):**
- `finalize_cycle(N)` speichert die Cycle-Metriken, die Konfusionsmatrix, und triggert den UMAP-Embedding-Bau.
- **Wichtig: Die UMAP für Zyklus N zeigt den Zustand *nach* Training von Zyklus N, aber die als „queried" markierten Punkte (pool=2) sind die Bilder, die *nach Zyklus N-1* abgefragt wurden.** Das ist ein Offset, den man in der Interpretation berücksichtigen muss.
- UMAP läuft in einem **Background-Thread** (`threading.Thread(target=_run_umap_background, daemon=True)`), damit der Haupt-Worker-Thread nicht blockiert wird. Der Pfad zur `.npz`-Datei wird sofort zurückgegeben (deterministischer Pfad), der Results-Tab liest die Datei erst, wenn sie geschrieben wurde.

**5. Querying (`query_samples` / `query_and_auto_annotate`):**
- Die Strategie-Funktion bekommt das aktuelle Modell und den Unlabeled-Pool-Loader. Sie gibt **relative Indizes** zurück (Position 0..N innerhalb des Unlabeled-Pools), die dann über `data_manager.unlabeled_to_absolute()` in absolute Dataset-Indizes konvertiert werden. Das ist eine kritische Stelle — relative Indizes direkt an `update_labeled_pool()` zu übergeben war ein Bug, der früh gefangen wurde.
- Es gibt **zwei Code-Pfade**: `query_samples()` (für Manual-Annotate) und `query_and_auto_annotate()` (der Fast-Path für Simulationsexperimente). Beide Pfade mussten separat mit Query-Summaries und UMAP-Evolution-Hooks versehen werden — ein Fehler war, nur `query_samples()` zu instrumentieren und zu übersehen, dass `query_and_auto_annotate()` der Pfad ist, der in echten Experimenten genutzt wird.

**6. Annotation & Pool-Update:**
- Bei Auto-Annotate: Ground-Truth-Labels werden direkt zugewiesen.
- Bei Manual-Annotate: Die UI zeigt eine Galerie mit den selektierten Bildern. Der Benutzer wählt Labels aus einer Dropdown-Liste. Submit sendet die Annotationen an den Controller, der sie über die Command-Queue an den Worker weitergibt. Ein **Query-Token** (UUID) stellt sicher, dass veraltete Annotationen (z.B. nach einem UI-Refresh) abgelehnt werden.

### Die vier Strategien — Was genau passiert im Code

Alle vier leben in `strategies.py` und haben die gleiche Signatur: `(model, unlabeled_loader, n_samples, device, heartbeat_fn) -> np.ndarray`.

**Random Sampling (Baseline):**
`np.random.choice(n_total, size=n_query, replace=False)` — fertig. Ignoriert das Modell komplett. Das ist die Baseline, gegen die alles verglichen wird.

**Least Confidence:**
Für jedes Bild im Unlabeled-Pool: Forward-Pass → Softmax → `max(probs)`. Dann: Die Bilder mit dem *niedrigsten* Maximum werden selektiert. Sortierung via `np.argsort(confidences)[:n_samples]`. Intuition: Wenn das Modell bei seiner Top-Vorhersage nur 5% Confidence hat, ist es maximal unsicher.

**Entropy:**
Für jedes Bild: Forward-Pass → Softmax → Shannon-Entropy: `H = -Σ(p · log(p))`. Sortierung: höchste Entropy zuerst (`np.argsort(-entropies)[:n_samples]`). Entropy berücksichtigt die *gesamte* Wahrscheinlichkeitsverteilung, nicht nur das Maximum. Bei 196 Klassen ist das ein entscheidender Unterschied — ein Modell kann niedrige Top-Confidence haben, aber trotzdem fast die gesamte Masse auf 3-4 Klassen konzentrieren (niedrige Entropy). Umgekehrt kann es die Masse gleichmäßig auf 50 Klassen verteilen (hohe Entropy). Entropy fängt diesen Unterschied ein, Least Confidence nicht.

**Margin Sampling:**
Für jedes Bild: Forward-Pass → Softmax → `top2 = topk(probs, k=2)` → `margin = top2[0] - top2[1]`. Kleiner Margin = das Modell schwankt zwischen genau zwei Klassen. Sortierung: kleinster Margin zuerst. Das ist besonders informativ, wenn das Modell bereits einige Klassen gut trennen kann und nur an spezifischen Grenzen unsicher ist.

**Warum Entropy theoretisch am besten für 196 Klassen sein sollte:**
Bei wenigen Klassen (z.B. 5) verhalten sich Entropy und Least Confidence fast identisch. Bei vielen Klassen divergieren sie, weil Entropy die volle Verteilung bewertet. Margin fokussiert auf genau zwei Klassen und ignoriert den Rest. Entropy maximiert den Informationsgewinn über den gesamten Hypothesenraum — das sollte bei feingranularer Klassifikation den größten Vorteil bringen.

### Query Summaries (Supervisor-Feature)

Nach jedem Query werden gespeichert: Klassenverteilung der selektierten Bilder, Unsicherheitsstatistiken (Min, Max, Mean, Std der Uncertainty-Scores), und welche Bilder selektiert wurden. Das zeigt, ob die Strategie systematische Biases hat — z.B. ob Entropy immer wieder dieselben schwierigen Klassen abfragt.

---

## Folie 4 — Technische Architektur (~4 Min.)

### Das fundamentale Streamlit-Problem

Streamlit führt bei **jeder Benutzerinteraktion das gesamte Skript von oben nach unten neu aus.** Klick auf einen Button → kompletter Rerun von `app.py`. Das bedeutet: Lokale Variablen verschwinden, Zustände gehen verloren, und — das Entscheidende — **Training, das Minuten oder Stunden dauert, kann nicht innerhalb eines Streamlit-Callbacks laufen.** Ein Button-Callback darf maximal Millisekunden dauern, sonst blockiert die gesamte UI.

### Die Lösung: Background-Thread + Event-System

Die Architektur besteht aus drei Schichten:

**1. Controller (Singleton via `@st.cache_resource`):**
Der Controller ist die einzige Instanz, die Streamlit-Reruns überlebt. `@st.cache_resource` ist ein Streamlit-Dekorator, der das Ergebnis einer Funktion prozessweit cached — egal wie oft das Skript neu ausgeführt wird, `get_controller()` gibt immer dieselbe Instanz zurück. Der Controller besitzt den `ExperimentState` und ist die **einzige Autorität für Zustandsübergänge** — `controller.dispatch()` mit Pattern-Matching (`match event.type: case ...`) behandelt alle Events.

**2. Worker-Thread (Daemon):**
Wenn der User auf „Start Experiment" klickt, spawnt der Controller einen Daemon-Thread (`worker.py: run_experiment()`). Dieser Thread baut den AL-Loop auf, trainiert, evaluiert, queried — und kommuniziert ausschließlich über Events. Der Worker schreibt **nie direkt** in den ExperimentState. Stattdessen emittiert er Events an die Inbox.

**3. Event-Inbox (Version-Counter-basiert):**
Die Inbox (`events.py`) ist eine thread-sichere Liste mit einem Versionszähler. Der Worker ruft `inbox.put(Event(...))` auf. Die UI-Seite prüft über `controller.process_inbox(last_version)`, ob neue Events vorliegen — wenn die Version sich nicht geändert hat, wird nichts getan. Das ist extrem günstig: ein Integer-Vergleich statt Queue-Draining.

### Zwei-Kanal-Kommunikation

Die Kommunikation ist bewusst asymmetrisch:

- **Worker → UI (Inbox):** Viele Events, kontinuierlich (EPOCH_DONE nach jeder Epoche, EVAL_COMPLETE, etc.). Verwendet die Inbox mit Versions-Counter.
- **UI → Worker (Command-Queue):** Seltene Befehle (STOP_EXPERIMENT, SUBMIT_ANNOTATIONS, NEXT_STEP). Verwendet eine einfache `queue.Queue`.

Warum nicht einfach eine Queue für alles? Weil die Inbox drain-by-version Semantik hat — die UI kann „hat sich etwas geändert?" für fast null Kosten prüfen. Eine Queue müsste man draining, und bei leerer Queue würde `get()` blockieren.

### Immutable Events

Jedes Event ist eine `@dataclass(frozen=True)` mit `MappingProxyType` für das Data-Dictionary. Warum? Weil der Worker und die UI in verschiedenen Threads laufen. Wenn der Worker ein Event emittiert und danach das `data`-Dict mutiert, sieht die UI inkonsistente Daten. Durch `frozen=True` + `copy.deepcopy()` + `MappingProxyType` im `__post_init__` ist das unmöglich. Kein Lock nötig für die Events selbst.

### Atomic Snapshots

Die UI ruft `controller.get_snapshot()` auf, das `ExperimentState.snapshot()` delegiert. Snapshot macht einen `deepcopy` aller Felder unter einem Lock. Die View bekommt damit einen konsistenten Zustand — keine Race Condition, bei der z.B. `current_cycle` schon aktualisiert ist, aber `epoch_metrics` noch vom alten Zyklus stammt.

### AppState-Zustandsmaschine

Die gesamte UI — welche View gerendert wird, welche Buttons aktiv sind, welches Polling-Intervall gilt — wird von einem einzigen `AppState`-Enum gesteuert:

```
IDLE → INITIALIZING → TRAINING → QUERYING → ANNOTATING → (nächster Zyklus)
                   ↓                              ↓
              WAITING_STEP                    FINISHED
                   ↓
              (NEXT_STEP-Befehl) → TRAINING
```

Jeder Übergang läuft durch `controller.dispatch()`. Es gibt **keinen Code**, der `app_state` direkt setzt, außer im Controller. Die Views prüfen nur `snap["app_state"]` und rendern entsprechend.

### Adaptive Polling

Streamlit's `@st.fragment(run_every=...)` erlaubt es, nur Teile der Seite periodisch zu aktualisieren. Ich verwende drei Modi:

- **Fast (0.5s):** Für QUERYING und ANNOTATING — hier passiert etwas, das der User sofort sehen soll.
- **Slow (1.5s):** Für TRAINING — Epochen dauern Sekunden, 1.5s-Updates reichen.
- **Off:** Für IDLE, FINISHED, ERROR — kein Polling nötig.

Der Moduswechsel passiert automatisch: `_ensure_poll_mode_matches_state()` vergleicht den gewünschten Modus mit dem aktuellen und triggert `st.rerun()` bei Abweichung.

**Entscheidende Optimierung:** Das Polling-Fragment rendert **nur den Main-Tab**. Results, Compare und Explorer werden nur bei vollem Page-Rerun aktualisiert. Sonst würde jeder Polling-Tick alle Tabs neu rendern — einschließlich der festplattenbasierten Results, die Dateien scannen.

### Weitere architektonische Patterns

**Heartbeat-Watchdog (120s Timeout):**
Der Worker aktualisiert `heartbeat_ts` bei jedem Event und in strategischen Schleifen (daher `heartbeat_fn` als Parameter in den Strategien). Wenn die UI sieht, dass der Heartbeat älter als 120s ist, zeigt sie eine Warnung — der Thread ist vermutlich blockiert oder abgestürzt.

**Query-Token (UUID pro Query):**
Jeder Query-Durchgang generiert eine UUID. Wenn der User Annotationen submitted, muss das Token übereinstimmen. Das verhindert, dass Annotationen von einem alten Query (z.B. nach einem Browser-Refresh, der den UI-State zurücksetzt) fälschlich akzeptiert werden.

**Incremental Persistence:**
Nach jedem Zyklus werden die Cycle-Metriken sofort als JSON gespeichert. Crash bei Zyklus 8? Zyklen 1–7 sind sicher auf der Festplatte. Der Results-Tab liest direkt vom Dateisystem.

**`num_workers=0` Enforcement auf Windows:**
PyTorch-DataLoader mit Multiprocessing inside eines Daemon-Threads deadlockt auf Windows. Auf Linux/Cluster ist es kein Problem (`num_workers=6` im Default-Config), aber auf Windows wird es erzwungen auf 0 gesetzt.

### Architektur als Thesis-Beitrag

Ein konkretes Ergebnis: Als der gesamte ML-Kern überarbeitet wurde (SupCon-Loss, diskriminative Lernraten, Reset-Mode-Fix, Embedding-Extraktion) — **Null Änderungen am Controller, Event-System oder den View-Dateien**. Die Schichtentrennung hat gehalten. Das ist explizit thesis-würdig als Software-Engineering-Beitrag.

---

## Folie 5 — ML-Kern (~5 Min.)

### Transfer Learning: Warum und Wie

196 Klassen mit einem initialen Pool von 500 Bildern — das sind ~2,5 Bilder pro Klasse. Von Grund auf trainieren ist unmöglich. Deshalb: Transfer Learning mit einem auf ImageNet vortrainierten Backbone.

**Modell-Loading:**
`timm.create_model(name, pretrained=True, num_classes=196)` — TIMM (PyTorch Image Models) liefert Hunderte vortrainierte Architekturen. Der `num_classes`-Parameter ersetzt automatisch den letzten Linear-Layer. Das Default-Modell ist ResNet-18, aber die Sidebar bietet eine kuratierte Auswahl (ResNet-Familie, EfficientNet, MobileNet).

**Freeze-then-Unfreeze-Strategie:**
Die ersten `freeze_backbone_epochs` Epochen (Default: 2) wird nur der Klassifikationskopf trainiert. Warum? Der Kopf hat zufällige Gewichte. Würde man sofort den gesamten Backbone mittrainieren, propagieren die riesigen Gradienten des zufälligen Kopfes rückwärts in den Backbone und zerstören die guten ImageNet-Features. Erst wenn der Kopf sinnvolle Gradienten liefert, wird der Backbone aufgetaut.

**Implementierungsdetail — Optimizer-Rebuild:**
Nach dem Unfreeze wird der Optimizer **komplett neu erstellt** (`self.optimizer = self._create_optimizer()`), nicht nur `requires_grad` umgeschaltet. Grund: Adam-Optimizer akkumuliert für jeden Parameter eine Schätzung des ersten und zweiten Moments (Laufender Mittelwert und Varianz der Gradienten). Backbone-Parameter hatten während des Einfrierens keinen Optimizer-State. Würde man den alten Optimizer behalten, hätten die frisch aufgetauten Parameter keine Momentum-Historie — die ersten Updates wären überdimensioniert. Neuer Optimizer = sauberer Start für alle Parameter.

Auch der Scheduler wird neu erstellt (`_create_scheduler(skip_warmup=True)`), damit die Warmup-Phase nicht doppelt läuft.

**Diskriminative Lernraten:**
Nach dem Unfreeze bekommt der Backbone eine reduzierte Lernrate: `backbone_lr = lr × backbone_lr_factor` (Default: 0.1×). Der Kopf bekommt die volle Lernrate. Das wird über separate Parameter-Gruppen im Optimizer realisiert:

```python
params = [
    {"params": backbone_params, "lr": backbone_lr},   # 1e-5
    {"params": head_params, "lr": lr},                  # 1e-4
]
```

Warum? Der Backbone hat bereits gute Features — er braucht nur leichte Anpassung (Fine-Tuning). Zu hohe Lernrate = Catastrophic Forgetting, die ImageNet-Features gehen verloren. Der Kopf muss schneller lernen, weil er komplett neu ist.

### Verlustfunktionen

Das System unterstützt drei Modi:

**1. CrossEntropy mit Label Smoothing (Default):**
`nn.CrossEntropyLoss(label_smoothing=0.1)` — statt 100% Wahrscheinlichkeit auf die Zielklasse wird die Verteilung geglättet: 90% auf die Zielklasse, die restlichen 10% gleichmäßig auf alle anderen. Warum ist das für Active Learning kritisch? AL-Strategien (außer Random) basieren auf Softmax-Wahrscheinlichkeiten. Ein overconfidentes Modell, das immer 99% auf eine Klasse gibt, hat niedrige Entropy überall — die Strategie sieht keine informativen Samples. Label Smoothing verhindert diese Überconfidenz und macht die Unsicherheits-Scores informativer.

**2. Supervised Contrastive Loss (SupCon, Khosla et al. 2020):**
Arbeitet im Embedding-Raum, nicht im Label-Raum. Zieht Embeddings derselben Klasse zusammen, drückt verschiedene Klassen auseinander. Die Implementierung in `losses.py`:

- Eine **Projection Head** (MLP: `Linear(feat_dim, 512) → ReLU → Linear(512, 128) → L2-Normalize`) projiziert die Backbone-Features in einen niedrigdimensionalen Raum. Nur für SupCon verwendet, wird bei Inferenz verworfen.
- **Similarity-Matrix:** Cosinus-Ähnlichkeit aller Paare, skaliert mit Temperatur τ=0.07. Niedrige Temperatur = schärfere Trennung.
- **Positive Mask:** Alle Paare mit gleichem Label (ohne Self-Paare), auf die der Zähler des Log-Softmax angewendet wird.
- **Numerische Stabilität:** `logits_max` wird abgezogen (LogSumExp-Trick), `ε=1e-12` verhindert Division durch Null.
- **Fallback:** Wenn kein Sample ein positives Paar hat (bei sehr kleinen Batches oder seltenen Klassen), gibt die Loss `torch.tensor(0.0, requires_grad=True)` zurück — kein Crash, aber auch kein Signal.

**3. Combined Loss:**
`loss = (1 - α) · CE + α · SupCon` — nutzt beide Signale. CE für direkte Klassifikation, SupCon für bessere Feature-Struktur. Die Kombination passiert mit **einem einzigen Forward-Pass** über einen `register_forward_hook` auf `model.global_pool`:

```python
hook_out = {}
def _hook(m, i, o):
    hook_out['feat'] = o
handle = self.model.global_pool.register_forward_hook(_hook)
outputs = self.model(images)          # Forward-Pass → logits + Features
handle.remove()
feats = hook_out['feat'].view(...)    # Features abgreifen
proj = self.projection_head(feats)     # Projection für SupCon
sc_loss = self.supcon_criterion(proj, labels)
ce_loss = self.criterion(outputs, labels)
loss = (1 - alpha) * ce_loss + alpha * sc_loss
```

Warum Hook statt zwei Forward-Passes? Effizienz — ein Forward-Pass durch ResNet-50 mit 224×224 Bildern kostet GPU-Zeit. Ein Hook auf `global_pool` fängt die Features ab, bevor der Klassifikationskopf sie bekommt. Das war ursprünglich ein Bug (Double Forward Pass), der behoben wurde.

### UMAP-Embeddings

UMAP (Uniform Manifold Approximation and Projection) wird verwendet, um die hochdimensionalen Backbone-Features (z.B. 512D für ResNet-18) auf 2D zu projizieren. Der Zweck ist rein visualisierungsorientiert — man will *sehen*, ob sich die Klassen im Feature-Raum trennen und wie sich diese Trennung über die AL-Zyklen entwickelt.

**Implementierung (`embeddings.py`):**
- Features werden über `trainer.get_embeddings()` extrahiert — Forward-Pass ohne Gradientenberechnung, Features vom `global_pool`-Layer.
- Der Unlabeled-Pool wird auf 2.000 Samples gecappt (`UMAP_UNLABELED_SAMPLE_LIMIT`), damit UMAP in akzeptabler Zeit läuft. Stanford Cars hat ~16K Bilder, aber Plotly `scattergl` kann 2K+ Punkte ohne Probleme darstellen. Datashader wurde bewusst verworfen.
- UMAP-Parameter: `n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42`. Cosinus-Metrik, weil die Features vor der UMAP nicht L2-normalisiert werden — Cosinus ist robust gegenüber unterschiedlichen Feature-Skalen.
- Jeder Punkt bekommt ein `pool_membership`-Label: 0=gelabelt, 1=ungelabelt, 2=queried (in diesem Zyklus abgefragt).

**Supervisor-Feature — UMAP-Evolution:**
Die gepunkteten "queried"-Markierungen (pool=2) zeigen, *welche Punkte der letzte Query selektiert hat*. Idealerweise liegen diese Punkte nahe an Entscheidungsgrenzen — dort, wo Cluster sich überlappen. Über die Zyklen sollte man sehen: Klassen-Cluster werden kompakter und besser getrennt, queried-Punkte bewegen sich von Cluster-Rändern weg, weil das Modell dort sicherer wird.

### Kalibrierung (ECE & Temperature Scaling)

**Expected Calibration Error (ECE):**
Wird nach jeder Evaluation berechnet. Methode: Alle Predictions werden nach Confidence in 15 Bins sortiert. Für jedes Bin: Vergleich zwischen durchschnittlicher Confidence und tatsächlicher Accuracy. Gewichteter Durchschnitt der Differenzen = ECE. Perfekt kalibriertes Modell: ECE = 0.

**Warum relevant für AL?**
Active Learning basiert auf der Annahme, dass Softmax-Wahrscheinlichkeiten sinnvolle Unsicherheitsschätzungen sind. Ein schlecht kalibriertes Modell (ECE hoch) liefert unzuverlässige Unsicherheitswerte → die Strategie selektiert suboptimale Bilder. ECE-Tracking über die Zyklen zeigt, ob mehr Daten die Kalibrierung verbessern — das ist ein Winkel, den die meisten AL-Paper übersehen.

**Temperature Scaling (Post-Hoc-Fix):**
Ein einzelner Skalar T wird gelernt, der die Logits vor dem Softmax teilt: `softmax(logits / T)`. T > 1 = weichere Verteilung (weniger overconfident), T < 1 = schärfere Verteilung. Optimiert wird T mit L-BFGS auf der NLL-Loss des Validierungssets (`trainer.py: calibrate_temperature()`). Das ist ein sauberer Thesis-Punkt: Raw-ECE vs. Temperature-Scaled-ECE pro Strategie vergleichen.

### Learning Rate Scheduling

Zwei unterstützte Scheduler:

- **Cosine Annealing mit Linear Warmup:** `SequentialLR([LinearLR(warmup), CosineAnnealingLR])`. Die Lernrate startet niedrig, steigt linear in den Warmup-Epochen, dann fällt sie sanft per Cosinus ab. Wird nach dem Backbone-Unfreeze **ohne Warmup** neu erstellt (`skip_warmup=True`), weil die Warmup-Phase nur einmal am Anfang nötig ist.
- **ReduceLROnPlateau:** Reduziert die LR, wenn die Val-Accuracy stagniert. Kein Warmup.

### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — begrenzt die Gradientennorm. Verhindert explodierende Gradienten, besonders in den ersten Epochen mit kleinem Pool und hoher Varianz in den Batches.

### Data Augmentation

Train-Transforms: `RandomResizedCrop(224) → RandomHorizontalFlip → RandomRotation(10°) → ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) → Normalize(ImageNet-Statistiken)`.

Val/Test-Transforms: `Resize(256) → CenterCrop(224) → Normalize`. Keine Augmentation — deterministisch für faire Evaluation.

Die ImageNet-Normalisierung (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) ist zwingend, weil die vortrainierten Gewichte auf diese Statistiken trainiert wurden.

---

## Folie 6 — Beispiel-Experimente

> **[PLATZHALTER — wird nach Abschluss der Experimentläufe ergänzt]**
>
> Geplante Inhalte:
> - Accuracy-Learning-Curves: Alle 4 Strategien über 10 Zyklen, X-Achse = gelabelte Samples, Y-Achse = Test-Accuracy
> - UMAP-Evolution: Snapshots von Zyklus 1, 5, 10 — zeigt wie sich die Embedding-Landschaft entwickelt
> - ECE-Verlauf über die Zyklen pro Strategie
> - Query-Verteilungsanalyse: Welche Klassen werden von welcher Strategie bevorzugt?
> - Konfusionsmatrix für besten und schlechtesten Zyklus
> - Probe-Images: 12 fixe Validierungsbilder, deren Vorhersage sich über die Zyklen entwickelt
>
> Experiment-Config:
> - Dataset: Stanford Cars (196 Klassen, ~16K Bilder)
> - Modell: ResNet-18, pretrained
> - Epochen pro Zyklus: 20, Warmup: 1, Freeze: 2
> - Initialer Pool: 500 (stratifiziert)
> - Query-Batch: 100 pro Zyklus
> - 10 Zyklen, auto_annotate=true, seed=42
> - 4 Runs: Entropy, Least Confidence, Margin, Random
> - Erwartete Laufzeit: ~6–7 Stunden für alle 4 Runs

---

## Folie 7 — Fazit (~2 Min.)

### Was wurde gebaut

Ein vollständig funktionsfähiges, interaktives Active-Learning-Framework, das als Forschungstool und als Lehrinstrument fungiert. Der gesamte AL-Zyklus — von der Konfiguration über das Training bis zur Evaluation — ist über ein Web-Dashboard steuerbar und visuell nachvollziehbar.

### Technische Beiträge

1. **Architektonische Entkopplung:** Saubere Trennung zwischen ML-Backend und Streamlit-Frontend über ein typisiertes Event-System. Beweis: Der gesamte ML-Kern (Losses, LR-Strategien, Reset-Modi, Embeddings) wurde mehrfach überarbeitet — ohne eine Zeile am Controller oder den Views zu ändern.

2. **Thread-Sicherheit in Streamlit:** Lösung für ein fundamentales Framework-Constraint (Script-Rerun bei jeder Interaktion) durch Singleton-Controller, Daemon-Worker-Thread, Immutable Events und Atomic Snapshots.

3. **Vollständige Experiment-Pipeline:** Nicht nur Training, sondern auch UMAP-Visualisierung, ECE-Kalibrierung, Query-Summaries, Probe-Images, Konfusionsmatrizen — alles automatisch pro Zyklus gespeichert und im Dashboard abrufbar.

### Lessons Learned

- **Epoch-Budget ist kritisch für SupCon:** Mit 5 Epochen pro Zyklus und 2 Warmup + 2 Freeze blieb nur 1 effektive Trainingsepoche — viel zu wenig. Erst mit 20 Epochen pro Zyklus konnte SupCon konvergieren.
- **Smoke-Tests sind keine Leistungstests:** ~20 Samples pro Zyklus auf einen ~100-Sample-Pool sind zu wenig, um den Feature-Raum von ResNet umzustrukturieren. UMAP-„Degradierung" in Smoke-Tests war kein Strategieversagen, sondern ein Skaleneffekt.
- **Beide AL-Pfade instrumentieren:** `query_and_auto_annotate()` ist der Pfad, der in echten Experimenten genutzt wird — nur `query_samples()` zu hooken verpasst die Simulation.
- **`reset_mode="pretrained"` zerstört Fortschritt:** Ein kritischer Bug — bei jedem Zyklus wurden die gelernten Gewichte verworfen und durch ImageNet-Weights ersetzt. Der Fix: `reset_mode="continue"` als Default, mit Freeze/Unfreeze und diskriminativen Lernraten.

### Ausblick

- **MC Dropout** (Monte Carlo Dropout) als alternative Unsicherheitsschätzung wurde bewusst auf die Discussion/Future-Work-Sektion verschoben — 20× Compute-Overhead würde alle Baselines ungültig machen.
- **Temperature Scaling** als Post-Hoc-Kalibrierungsfix ist niedrig-riskant und produziert einen sauberen Thesis-Vergleich (Raw-ECE vs. Scaled-ECE).
- Das Framework ist erweiterbar: Neue Strategien brauchen nur eine Funktion mit der richtigen Signatur + Eintrag im Registry-Dict. Neue Events: Enum-Variante + Worker-Emit + Controller-Case.

## Experiments

> **Dauer:** ~3–4 Minuten gesamt

---

### Folie 1 — Lernkurven-Chart (4 Strategien)

> „Ich zeige euch jetzt die ersten Experimente, die ich mit dem System durchgeführt habe.
>
> Wir haben vier Active-Learning-Strategien miteinander verglichen: Entropy Sampling, Least Confidence, Margin Sampling — und als Baseline einfaches Random Sampling.
>
> Das Setup war für alle vier identisch: ResNet50 als vortrainiertes Modell, der Stanford Cars Datensatz mit 196 Fahrzeugklassen, und 30 Active-Learning-Zyklen. Wir starten mit 1.000 beschrifteten Bildern und fragen pro Zyklus 100 neue an — am Ende also 3.900 Labels insgesamt.
>
> Was ihr hier seht, ist die Testgenauigkeit über die Anzahl der beschrifteten Bilder. Alle vier Kurven steigen — das System lernt, Zyklus für Zyklus, mit jedem neuen Label dazu.
>
> Am Ende landen alle Strategien bei rund 52 bis 53 Prozent Testgenauigkeit. Das klingt auf den ersten Blick nicht viel — aber bei 196 teils sehr ähnlichen Fahrzeugklassen ist das ein durchaus solides Ergebnis für diesen Labelumfang."

**Screenshot:** Compare-Tab → „Test Accuracy" — alle 4 Runs ausgewählt

---

### Folie 1 — Key Insights

> „Was interessanter ist als die finale Genauigkeit: das Verhalten in den frühen Zyklen.
>
> Entropy Sampling ist dort konsistent vorne — das bedeutet, es braucht weniger Labels, um den gleichen Genauigkeitspunkt zu erreichen. Genau das ist der Kerngedanke von Active Learning: mit weniger Daten mehr herausholen.
>
> Besonders auffällig ist Margin Sampling: In den Zyklen 4 bis 9 stagniert die Kurve fast komplett. Das liegt daran, dass Margin Sampling auf den Abstand zwischen den zwei wahrscheinlichsten Klassen schaut — wenn das Modell aber noch schlecht kalibriert ist, ist dieses Signal reines Rauschen. Das Modell wählt Bilder aus, die ihm zufällig schwer vorkommen, aber nicht wirklich informativ sind. Das ist ein bekannter Kaltstart-Effekt, den wir hier empirisch beobachten konnten.
>
> Interessant ist auch: Random Sampling ist erstaunlich schwer zu schlagen. Das zeigt, dass bei einem so vielfältigen Datensatz fast jedes zufällig gewählte Bild eine gewisse Information trägt."

**Screenshot:** Compare-Tab → „Final Results Summary" Tabelle (rechts unten auf der Folie)

---

### Folie 2 — Was das System demonstriert

> „Diese Experimente zeigen vor allem eines: das Framework funktioniert. Vier verschiedene Strategien, jeweils 30 Zyklen, vollständig automatisiert durchgelaufen — mit einheitlichem Setup, reproduzierbarer Konfiguration und sauber gespeicherten Ergebnissen pro Lauf.
>
> Das System ist in der Lage, solche Vergleichsexperimente strukturiert und nachvollziehbar durchzuführen — was die Grundlage für aussagekräftige Evaluierungen ist."

**Screenshots:**
- Results-Tab → UMAP Evolution, Side-by-side: Zyklus 1 vs. Zyklus 30 (Pool Membership)
- Results-Tab → „Confusion by body type" Heatmap (letzter Zyklus)

---

### Folie 2 — Nächste Schritte

> „Gleichzeitig sehen wir klar, wo die nächsten Experimente ansetzen müssen.
>
> Erstens fehlt uns noch eine vollüberwachte Obergrenze — also ein Modell, das auf allen verfügbaren Daten trainiert wird. Erst dann können wir sagen, wie viel Dateneinsparung Active Learning wirklich bringt.
>
> Zweitens trainieren wir derzeit jeden Zyklus genau 20 Epochen — egal wie viele Daten vorhanden sind. In frühen Zyklen ist das zu viel, in späteren zu wenig. Mit echtem Early Stopping wird jeder Zyklus fairer.
>
> Drittens haben wir bisher nur einen einzigen Seed verwendet. Für statistisch belastbare Aussagen brauchen wir mehrere Durchläufe.
>
> Und schließlich möchte ich eine diversitätsbasierte Strategie wie Core-Set testen — die nicht nur auf Unsicherheit schaut, sondern auch darauf, dass die ausgewählten Bilder sich möglichst unterscheiden. Das könnte gerade in frühen Zyklen die Schwäche von Margin Sampling ausgleichen.
>
> Die Infrastruktur steht — die nächsten Experimente werden deutlich aussagekräftiger sein."
---

## Timing-Empfehlung

| Folie | Minuten | Kumuliert |
|-------|---------|-----------|
| 1 — Motivation & Problemstellung | 3 | 3 |
| 2 — System-Überblick & Demo | 3 | 6 |
| 3 — Active Learning Pipeline | 4 | 10 |
| 4 — Technische Architektur | 4 | 14 |
| 5 — ML-Kern | 5 | 19 |
| 6 — Beispiel-Experimente | 4 | 23 |
| 7 — Fazit | 2 | 25 |

> **Tipp für die Demo:** Wenn möglich, die Quick-Test-Config (`quick_test.yaml`, 4 Klassen, 3 Zyklen) vorab starten und den fertigen Run zeigen. Live-Training kann schiefgehen. Alternativ: Screenshots von einem abgeschlossenen Run vorbereiten.
