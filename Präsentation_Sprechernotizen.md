# Sprechernotizen — Praxisprojekt-Präsentation (20 min)

> **Lese-Hinweise**
> - Sprechtempo-Annahme: ~120 Wörter/Minute. Wer langsam spricht, kürzt jeweils den letzten Satz pro Folie.
> - "Warum-Anker" sind Stichpunkte für Q&A bzw. wenn man den roten Faden verliert — nicht vorlesen.
> - "Brücke" ist der Übergangssatz zur nächsten Folie. Hilft beim Folienwechsel ohne Pause.

---

## Folie 1 — Titel  (≈ 0:20)

**Worum es geht:** Begrüßung, Titel, Zeitrahmen.

**Sprechtext:**
Sehr geehrte Damen und Herren, ich darf Ihnen heute mein Praxisprojekt vorstellen: die Entwicklung einer interaktiven Active-Learning-Pipeline für die visuelle Analyse und Steuerung von Trainingsprozessen in der Fahrzeugklassifikation. In den nächsten 20 Minuten zeige ich Ihnen nicht nur, *was* gebaut wurde, sondern vor allem, *warum* es so gebaut wurde.

**Brücke:** Hier zunächst der Überblick.

---

## Folie 2 — Inhaltsverzeichnis  (≈ 0:30)

**Sprechtext:**
Wir starten mit Motivation und Problemstellung, gefolgt von einem System-Überblick und einer kurzen Live-Demo. Anschließend gehe ich auf die Active-Learning-Pipeline und ihre vier Strategien ein, danach auf die technische Architektur — also das Streamlit-Threading-Modell — und auf den ML-Kern: Transfer Learning, Verlustfunktionen, UMAP und Kalibrierung. Den Abschluss bilden Ergebnisse und Fazit.

**Brücke:** Beginnen wir mit der Motivation.

---

## Folie 3 — Motivation & Problemstellung  (≈ 1:30)

**Worum es geht:** Warum AL überhaupt? Warum fine-grained als Treiber?

**Sprechtext:**
Computer-Vision-Modelle brauchen viele gelabelte Daten — und genau das ist heute das Bottleneck, nicht die Rechenleistung und nicht die Architekturen. Bei der Fahrzeugklassifikation wird das doppelt schwer: wir haben es mit Fine-grained Visual Recognition zu tun. Klassen unterscheiden sich oft nur in winzigen Details — der Form eines Kühlergrills, der C-Säule, eines Heckleuchten-Ausschnitts. Das heißt: wir können nicht beliebig Crowdworker für Cents pro Bild engagieren, sondern brauchen Domänenwissen oder zumindest sehr saubere Referenzen. Manuelles Labeln ist also teuer, langsam und skaliert schlecht.

Der naheliegende Weg wäre, einfach mehr Geld in Annotation zu stecken. Das löst das Problem aber nicht — es verschiebt es nur. Die eigentliche Frage lautet: ist jedes Bild gleich informativ? Offensichtlich nicht. Ein Modell, das zwei Honda Civics aus 2010 schon gut trennen kann, lernt aus dem dritten Civic kaum etwas dazu. Und genau hier setzt die Kernfrage des Projekts an: kann das Modell selbst entscheiden, welche Bilder annotiert werden sollen — und mit einem Bruchteil der Labels eine vergleichbare Genauigkeit erreichen? Das ist exakt das Versprechen von Active Learning.

**Warum-Anker:**
- AL statt "mehr labeln": dieselben Kosten lösen das Skalierungsproblem nicht, sie verschieben es.
- Fine-grained als Treiber: rechtfertigt überhaupt erst den Aufwand für Domain-Experten.
- "Informationsgehalt nicht gleichverteilt" ist die formale Grundannahme von AL.

**Brücke:** Wie schwer fine-grained tatsächlich ist, sieht man am besten an einem Beispiel.

---

## Folie 4 — Beispiel: BMW Sedan vs. Coupé  (≈ 0:45)

**Sprechtext:**
Hier sehen Sie zwei Fahrzeuge aus dem Stanford-Cars-Datensatz: links ein BMW 3er Sedan, Baujahr 2012, rechts ein BMW 3er Coupé, ebenfalls 2012. Gleicher Hersteller, gleiches Modell, gleicher Baujahr — und für unser Klassifikationssystem trotzdem zwei separate Klassen. Das einzige zuverlässige Unterscheidungsmerkmal sind die Anzahl der Türen und die Dachlinie. Bei 196 solcher Klassen wird klar, warum zufälliges Labeln Budget verschwendet: die meisten Bilder sind aus Modellsicht redundant, und nur eine Minderheit liegt auf den Klassengrenzen, wo das Modell wirklich Hilfe braucht.

**Warum-Anker:**
- Beispiel ist bewusst aus Stanford Cars — das ist der Datensatz, mit dem ich gemessen habe.
- Sedan/Coupé zeigt: das Problem ist nicht "verschiedene Marken erkennen", sondern "innerhalb derselben Modellfamilie unterscheiden".

**Brücke:** Genau dort soll Active Learning ansetzen — dafür brauchen wir aber ein Tool, das den Zyklus sichtbar macht.

---

## Folie 5 — System-Überblick  (≈ 1:30)

**Worum es geht:** Was sieht der Nutzer? Warum diese Aufteilung?

**Sprechtext:**
Das Framework besteht aus zwei Bereichen: einer Sidebar mit der gesamten Konfiguration und einer Laufzeit-Ansicht mit fünf Tabs.

In der Sidebar konfiguriere ich Modell — also ResNet, MobileNet oder EfficientNet — die Sampling-Strategie sowie sämtliche Hyperparameter: Epochenzahl, Lernrate, Scheduler, Freeze-Epochen, Loss-Funktion. Außerdem gibt es einen Step-Mode, in dem ich jeden Zyklus einzeln auslöse — das ist unschätzbar wertvoll, wenn man die Pipeline didaktisch erklären will.

Ich habe die UI bewusst in Tabs aufgeteilt und nicht auf eine Seite gelegt. Streamlit rendert bei jedem Klick das gesamte Skript neu — eine Monolith-Seite würde bei jedem Event alle Plots, Tabellen und Bilder neu erzeugen, das wäre praktisch unbenutzbar. Mit Tabs lade ich nur das, was gerade sichtbar ist.

Die Tabs sind: Main mit Live-Loss und Accuracy, Annotation mit den queried Bildern, Results mit Accuracy/F1/ECE über Zyklen sowie UMAP-Embeddings und Konfusionsmatrix, Compare zum direkten Vergleich mehrerer Experimente, und schließlich der Dataset Explorer für die Pool-Inhalte.

**Warum-Anker:**
- 5 Tabs statt 1 Seite: Streamlit re-rendert komplett bei jedem Klick, Tabs lazy-loaden Inhalt.
- Sidebar separiert: Konfiguration ist über alle Phasen hinweg gleich, Tab-Inhalte ändern sich.
- Step-Mode zentral für Didaktik — sonst rauscht der Zyklus durch und niemand sieht, was passiert.

**Brücke:** Damit Sie sehen, was ich meine — eine kurze Live-Demo.

---

## Folie 6 — Live-Demo  (≈ 3:00, Stichworte statt Volltext)

> Reines Drehbuch — wird gesprochen, während die Demo läuft. Kein vorgelesener Text.

- **Vor dem Klick:** Sidebar zeigen, hervorheben: ResNet-50 + Stanford Cars + Strategie Entropy + Step-Mode aktiv.
- **Start klicken** → Übergang in `INITIALIZING` kurz benennen ("hier baut der Worker-Thread gerade Pool und Modell auf").
- **Main-Tab** → Live-Loss-Kurve auftauchen lassen; ein, zwei Epochen, dann auf "Next Step" klicken.
- **Annotation-Tab** → queried Bilder mit Uncertainty-Werten zeigen; eine Zeile zur Farbcodierung sagen ("rot = Modell sehr unsicher").
- **Submit** mit Auto-Annotate (also GT-Labels) — Kommentar: "in der Bachelorarbeit ginge hier ein echter Annotator ans Werk".
- **Results-Tab nach 2–3 Zyklen** → Accuracy-Kurve, dann auf UMAP scrollen. Pool-Encoding kurz erklären (0=labeled, 1=unlabeled, 2=in diesem Zyklus gequeried).
- **Compare-Tab** kurz andeuten ("hier vergleiche ich später Strategien direkt").
- **Fallback,** falls etwas hakt: Dataset Explorer öffnen — der lädt immer schnell und zeigt die Pool-Verteilung.

**Brücke:** So weit das, was Sie als Nutzer sehen — werfen wir einen Blick darauf, was unter der Haube passiert.

---

## Folie 7 — AL-Pipeline und Index-Pools  (≈ 1:30)

**Worum es geht:** Der Zyklus + die nicht-offensichtliche technische Entscheidung dahinter.

**Sprechtext:**
Der Active-Learning-Zyklus besteht aus sechs Schritten: Stratified Init, Training, Evaluation, Query, Annotation, Pool-Update — und dann beginnt der nächste Zyklus.

Eine technische Besonderheit, die ich besonders hervorheben möchte: die Pools sind index-basiert. `_labeled_list` und `_unlabeled_list` sind reine Integer-Listen. Beim Pool-Update verschiebe ich nur Indizes — keine Bytes, keine Tensors, keine Datei-Operationen. Wir hätten Bilder zwischen zwei Verzeichnissen kopieren können, das wäre die naive Lösung — aber das skaliert nicht: bei 16 000 Bildern wären das pro Zyklus mehrere hundert Megabyte unnötiger I/O. So bleibt der Memory-Footprint konstant, egal wie groß der Datensatz wird.

Die Initialisierung ist stratifiziert mit einem Sample pro Klasse. Bei 196 Klassen sind das 196 Bilder im Initial-Pool. Der Grund: ein Cold-Start, bei dem manche Klassen gar nicht vorkommen, würde das Modell früh in eine schiefe Verteilung drücken — die fehlenden Klassen bekäme es später nur durch Zufall der Strategie zu sehen.

`reset_mode='continue'` als Default bedeutet: zwischen Zyklen behalten wir die Gewichte, setzen aber Optimizer-State und LR-Scheduler zurück. Frische Optimizer-Momente auf einem gewachsenen Pool — das verhindert, dass alte Adam-Momente das neue Datenregime überschatten.

**Warum-Anker:**
- Index-Pools: vermeiden Datenkopien → konstanter Speicher, schnelles Pool-Update.
- Stratified Init mit 1/Klasse: Cold-Start-Schutz; ohne das verzerren fehlende Klassen die ersten Zyklen.
- `reset_mode='continue'`: keine Lernfortschritte verwerfen, aber stale Optimizer-Momentum auch nicht reinschleppen.

**Brücke:** Die zentrale Frage bleibt: wie wählt das Modell die Top-k aus?

---

## Folie 8 — Die 4 Query-Strategien  (≈ 1:30)

**Worum es geht:** Welche Strategien, warum diese, und warum erwarte ich Entropy als Sieger?

**Sprechtext:**
Wir haben vier Strategien implementiert. Drei sind klassische Uncertainty-Maße, die vierte ist Random als Baseline.

Least Confidence schaut nur auf die Top-1-Klasse: 1 minus die Maximalwahrscheinlichkeit. Margin betrachtet Top-1 minus Top-2 — die Frage ist nicht "wie sicher", sondern "wie nah am Konkurrenten". Entropy hingegen nutzt die gesamte Posterior-Verteilung — die Summe minus P log P über alle 196 Klassen.

Warum genau diese drei und nicht etwa Query-by-Committee oder Diversity-Sampling? Erstens: alle drei sind günstig — ein einziger Forward-Pass über den Unlabeled-Pool reicht. Zweitens sind sie theoretisch klar trennbar, das macht den Vergleich aussagekräftig.

Bei 196 Klassen erwarten wir, dass Entropy gewinnt. Least Confidence und Margin reduzieren die ganze Verteilung auf ein oder zwei Werte — sie ignorieren, ob die restliche Wahrscheinlichkeitsmasse auf zwei oder auf zwanzig Klassen verteilt ist. Bei feinkörniger Klassifikation ist genau diese Streuung das informative Signal.

Random ist absichtlich dabei. Ohne Baseline können wir nicht behaupten, dass eine Strategie tatsächlich Information liefert — wir würden vielleicht nur den Sampling-Bias unseres Datensatzes messen.

**Warum-Anker:**
- Drei Uncertainty-Maße: günstig (1 Forward-Pass), theoretisch trennbar.
- QBC / Diversity bewusst weggelassen: kostet zusätzliche Modelle bzw. Distanzberechnungen, hätte den Scope gesprengt.
- Random behalten: ohne Nullhypothese ist jeder Vorsprung wertlos.

**Brücke:** Vier Strategien zu implementieren ist einfach — sie live und ohne UI-Freeze laufen zu lassen, ist die eigentliche Architektur-Aufgabe.

---

## Folie 9 — Streamlit-Constraints und Threading-Modell  (≈ 1:30)

**Worum es geht:** Die vier Streamlit-Probleme und je eine bewusste Lösung.

**Sprechtext:**
Streamlit ist von Haus aus zustandslos: bei jedem User-Klick wird das ganze Skript neu ausgeführt. Das bricht eine naive ML-Pipeline an vier Stellen, und für jede haben wir bewusst eine Lösung gewählt.

Erstens: ein synchron im Hauptthread laufendes Training würde die UI blockieren — keine Live-Plots, kein Stop-Button. Wir lagern das Training in einen Daemon-Thread aus.

Zweitens: ein Worker-Thread, der bei jedem Reload neu erzeugt wird, würde sofort sterben und neu starten. Wir lösen das mit `@st.cache_resource` — der Controller wird einmal pro Prozess instanziiert und überlebt alle Reloads. Eine globale Variable hätte zwar funktioniert, würde Streamlits Session-Modell aber brechen, sobald mehrere Tabs offen sind.

Drittens: zwei Threads, die gemeinsamen Mutable State teilen, sind ein Race-Condition-Generator. Alle Writes laufen daher unter einem `threading.Lock`, und die UI liest ausschließlich Snapshots — Deep-Copies, kein direkter Zugriff auf Worker-Felder.

Viertens: zwei Kanäle, nicht einer. Eine Event-Inbox vom Worker zur UI mit Versions-Counter, und eine Command-Queue von der UI zum Worker mit nicht-blockierendem `get_nowait`. Ein gemeinsamer Kanal hätte Priorisierung erzwungen, die wir nicht brauchen — und er hätte den UI-Thread potentiell blockieren können.

**Warum-Anker:**
- Daemon-Thread: trennt Training von UI.
- `@st.cache_resource`: Singleton, der Reloads überlebt, ohne Sessions zu sprengen.
- Lock + Snapshots: keine Race Conditions, klare Ownership.
- Zwei Kanäle: keine Priorisierung nötig, kein Blocking-Risiko.

**Brücke:** So weit die Streamlit-Seite — werfen wir einen Blick auf den ML-Kern.

---

## Folie 10 — Transfer Learning und Fine-Tuning  (≈ 1:30)

**Worum es geht:** Wie lade und trainiere ich die Modelle — und warum genau so?

**Sprechtext:**
Wir laden alle Modelle über TIMM. Torchvision wäre die offensichtliche Alternative gewesen, hat aber nur ein paar Dutzend Architekturen, und die API unterscheidet sich zwischen Modellen. TIMM bietet über tausend Modelle hinter einer einheitlichen API — und vor allem: `model.num_features` liefert für jedes Modell die Feature-Dimension, ohne dass ich das hartcodieren muss. Das macht den Loader generisch.

Ein Forward-Hook auf `model.global_pool` gibt mir die Embeddings, ohne die Architektur anfassen zu müssen. Das ist nicht nur eleganter — es bedeutet auch, dass ich neue Modelle hinzufügen kann, ohne den Embedding-Code zu ändern.

Beim Fine-Tuning friere ich den Backbone in den ersten Epochen ein. Der Grund: der Head ist zufällig initialisiert. Würde ich beide gleichzeitig trainieren, würden zufällig große Gradienten den vorhandenen ImageNet-Features schaden, bevor der Head überhaupt sinnvoll vorhersagt. Erst wenn der Head warmgelaufen ist, taue ich den Backbone auf.

Discriminative Learning Rates: der Backbone bekommt die 0,1-fache Head-Lernrate. Begründung: ImageNet-Features sind bereits gut, sie brauchen nur leichte Anpassungen ans Domänenproblem. Der Head muss ganz neue Klassen lernen — ihm tut die volle Lernrate gut.

**Warum-Anker:**
- TIMM statt torchvision: einheitliche API, `num_features` automatisch.
- Forward-Hook statt Architektur-Patch: keine Modell-Änderung nötig, generisch über alle Backbones.
- Freeze-Phase: schützt pretrained Features vor Random-Head-Gradienten.
- Discriminative LR: Backbone konservativ, Head aggressiv — passt zum Wissensstand der jeweiligen Schicht.

**Brücke:** Eine weitere Designentscheidung betrifft den Loss.

---

## Folie 11 — Verlustfunktionen  (≈ 1:30)

**Worum es geht:** Drei Loss-Modi, und insbesondere warum wir SupCon mit ins Boot holen.

**Sprechtext:**
Wir haben drei Loss-Modi. `cross_entropy` ist der Standard mit Label Smoothing — schnell, stabil, gut verstanden. `supcon` ist der reine Supervised Contrastive Loss — er optimiert nicht die Klassengrenzen, sondern die Feature-Geometrie: gleiche Klasse näher zusammen, andere Klassen weiter weg, im gesamten Batch.

Warum SupCon zusätzlich zu CE? Bei Fine-grained Recognition entscheidet der Embedding-Raum darüber, ob zwei sehr ähnliche Klassen überhaupt trennbar sind. CE optimiert nur die letzte Schicht — wenn der Embedding-Raum ungeeignet ist, kämpft die Softmax-Decision-Boundary auf verlorenem Posten. SupCon arbeitet eine Ebene tiefer.

Warum dann nicht einfach SupCon allein? Weil SupCon kein Klassifikator ist. Wir bekämen schöne Embeddings, aber für Active Learning brauche ich kalibrierte Klassen-Posteriors für die Entropy-Berechnung. Die Kombination `(1−α)·CE + α·SupCon` trainiert beide Ziele in einem einzigen Forward-Pass.

Der ProjectionHead — eine zwei-schichtige MLP — wird ausschließlich für SupCon benutzt, nicht für Query oder Evaluation. SupCon profitiert nachweislich von einem niedrig-dimensionalen Projektionsraum, aber genau dieser Raum verliert die klasseninformativen Details, die ich für Klassifikation und Embedding-Visualisierung brauche.

**Warum-Anker:**
- SupCon zu CE addieren: feinkörnige Trennung passiert im Embedding-Raum, nicht an der Decision-Boundary.
- Kombiniert statt SupCon-only: AL braucht kalibrierte Posteriors, die SupCon allein nicht liefert.
- ProjectionHead nur fürs Loss-Berechnen: Projektionsraum ist gut für Kontrastlernen, aber zu stark abstrahiert für Klassifikation und UMAP.

**Brücke:** Wenn wir schon bei Embeddings sind — UMAP und Kalibrierung sind die letzten beiden ML-Bausteine.

---

## Folie 12 — UMAP und Temperature Scaling  (≈ 1:30)

**Worum es geht:** Zwei orthogonale Probleme, zwei Lösungen.

**Sprechtext:**
Temperature Scaling adressiert ein bekanntes Phänomen: moderne neuronale Netze sind systematisch überkonfident. Eine Vorhersage mit Softmax 0,9 stimmt empirisch oft nur in 70 Prozent der Fälle. Für AL ist das fatal — Entropy würde "überschätzt sichere" Samples zu früh als wenig informativ einstufen. Wir lernen post-hoc einen einzigen Skalar T, optimiert mit LBFGS auf dem Validation-Set. Kein Retraining, ein Parameter — und die Posterior-Verteilung wird kalibrierter. Gemessen wird das mit ECE, dem Expected Calibration Error, vor und nach Scaling. LBFGS habe ich gewählt, weil das Problem konvex und niedrig-dimensional ist; SGD wäre Overkill.

UMAP-Embeddings dienen der Visualisierung. Ich hätte t-SNE nehmen können, aber t-SNE bewahrt nur lokale Strukturen — globale Cluster-Beziehungen gehen verloren, was bei 196 Klassen problematisch wird. PCA wäre schnell, ist aber linear und zeigt bei tief erlernten Embeddings kaum Struktur. UMAP bewahrt beides und läuft schnell genug.

Wir cappen den Unlabeled-Pool auf 2 000 Samples — UMAP skaliert mit O(N log N), das reicht für eine flüssige UI ohne sichtbaren Strukturverlust. Die Berechnung läuft im Daemon-Thread, damit der AL-Zyklus nicht blockiert; die UI zeigt die UMAP, sobald sie fertig ist.

**Warum-Anker:**
- Temperature Scaling statt Retraining: ein Parameter, fünf Sekunden Optimierung — Retraining wäre extrem teuer für denselben Effekt.
- LBFGS: konvexes 1-D-Problem, deterministisch, schnell.
- UMAP statt t-SNE: globale Struktur erhalten; statt PCA: nicht-linear.
- 2 000-Cap: O(N log N) bleibt subjektiv "instant" — bei 16 000 Punkten würden Nutzer sekundenlang warten.
- Daemon-Thread für UMAP: blockiert nicht den nächsten AL-Zyklus.

**Brücke:** Damit sind wir bei den Experimenten.

---

## Folie 13 — Experiment-Setup  (≈ 1:00)

**Sprechtext:**
Stanford Cars, 196 Klassen, offizieller Train-Test-Split. Modell ist ResNet-50, ImageNet-pretrained. ResNet-50 bietet aus meiner Sicht den besten Kompromiss aus Kapazität, Trainingszeit und Reproduzierbarkeit — ResNet-18 wäre für 196 fine-grained Klassen unterdimensioniert, ResNet-152 würde nur Trainingszeit kosten, ohne entsprechenden Genauigkeitsgewinn auf einem Datensatz dieser Größe.

Initial-Pool 500, Query-Batch 100 pro Zyklus, 30 Zyklen. Pro Zyklus 20 Epochen, kombinierter Loss, Seed 42. Hardware: NVIDIA H100 PCIe auf dem JupyterHub-Cluster der Hochschule.

Verglichen wird mit einer voll-überwachten Referenz auf dem gesamten Trainings-Split — 200 Epochen, sonst bewusst identische Konfiguration, damit der Vergleich fair bleibt.

**Warum-Anker:**
- ResNet-50 als Sweet Spot: groß genug für 196 Klassen, klein genug für 30 Zyklen Trainingsbudget.
- 30 Zyklen × 100 Samples: ergibt eine echte Lernkurve, ohne den vollen Datensatz auszuschöpfen.
- 200 Epochen Supervised: genug, dass die Baseline ihre Plateau-Accuracy erreicht — andernfalls wäre der Vergleich unfair zugunsten von AL.
- Identische Konfiguration sonst: alle anderen Variablen kontrolliert.

**Brücke:** Was kommt dabei heraus?

---

## Folie 14 / 15 — Ergebnisse  (≈ 2:30)

**Worum es geht:** Drei Hauptaussagen, mit Erklärung des Mechanismus dahinter.

**Sprechtext:**
Drei Hauptergebnisse.

Erstens: die Reihenfolge der Strategien ist Entropy vor Margin vor Least Confidence vor Random. Das ist genau, was wir bei 196 Klassen erwartet haben. Least Confidence und Margin destillieren die ganze Verteilung auf einen oder zwei Werte — bei wenigen Klassen ist das ausreichend, aber bei 196 verlieren wir Information darüber, wie die restliche Wahrscheinlichkeitsmasse verteilt ist. Entropy nutzt alle Klassen, und genau das zahlt sich hier aus.

Zweitens: AL-Strategien erreichen mit 56 Prozent der Labels rund 68 Prozent der voll-überwachten Accuracy. Das ist die zentrale Aussage des Projekts. Wir bezahlen Performance für Annotation-Effizienz, aber das Verhältnis ist deutlich besser als linear.

Drittens, und das ist mein Lieblingsergebnis: die AL-Läufe sind besser kalibriert als die Supervised-Baseline. In der Tabelle steht der unkalibrierte ECE, also vor Temperature Scaling: 0,16 für AL gegenüber 0,19 für Supervised. Das wirkt zunächst kontraintuitiv — weniger Daten, bessere Kalibrierung? Die Erklärung ist der Pool-Effekt: kleinere Trainingspools führen zu weniger Overfitting und damit zu von Haus aus weniger overkonfidenten Posteriors — schon bevor TS überhaupt greift. Temperature Scaling sitzt anschließend obendrauf und reduziert die ECE in beiden Fällen weiter, aber die Ausgangslage ist bei AL bereits günstiger. Praktisch bedeutet das: ein AL-trainiertes Modell ist schon ohne Nachkalibrierung näher an ehrlichen Wahrscheinlichkeiten.

Diese drei Ergebnisse stützen das Versprechen von Active Learning: wir müssen nicht zwischen Daten-Effizienz und verlässlicher Unsicherheit wählen. Beides gleichzeitig ist möglich — und für sicherheitskritische Anwendungen, etwa autonomes Fahren, ist die Kalibrierung manchmal sogar wichtiger als die rohe Accuracy.

**Warum-Anker:**
- Entropy gewinnt bei 196 Klassen: alle Klassen werden bewertet, nicht nur Top-1/Top-2.
- 56% Labels → 68% Accuracy: super-linearer Gewinn rechtfertigt den AL-Aufwand.
- AL besser kalibriert: kleinere Pools = weniger Overfit = von Haus aus weniger Overconfidence (gezeigte ECE-Werte sind pre-TS).
- Argument für sicherheitskritische Anwendungen: Kalibrierung > rohe Accuracy.

**Brücke:** Damit komme ich zum Fazit.

---

## Folie 16 — Fazit und Thesis-Beiträge  (≈ 0:45)

**Sprechtext:**
Drei Beiträge.

Empirisch: Uncertainty-Sampling schlägt Random auf Stanford Cars, und AL-Läufe sind besser kalibriert als die Supervised-Baseline.

Architektonisch: das entkoppelte Zwei-Thread-Design hat es ermöglicht, SupCon, UMAP und ECE nachträglich zu integrieren, ohne eine einzige Zeile UI-Code anzufassen — das ist nicht nur Codequalität, das ist die Voraussetzung dafür, dass die Pipeline auch im Rahmen meiner Bachelorarbeit weiterwachsen kann.

Und didaktisch: das interaktive Tool macht den AL-Zyklus, die Unsicherheitsmaße und die Kalibrierung visuell nachvollziehbar — etwas, das ein statisches Paper nicht leisten kann.

Vielen Dank für Ihre Aufmerksamkeit. Ich freue mich auf Ihre Fragen.

**Warum-Anker:**
- Drei separate Beiträge, weil jeder eigenständig: empirisch wäre auch ohne UI gültig, architektonisch wäre auch ohne SupCon-Ergebnisse interessant, didaktisch gilt unabhängig vom Datensatz.

---

## Backup-Antworten für Q&A

| Frage | Kurzantwort |
|------|-------------|
| Warum nicht Query-by-Committee? | Hätte n-fache Trainingszeit gekostet; Scope des Praxisprojekts. Implementierung in Bachelorarbeit denkbar. |
| Warum nicht MC-Dropout für Bayesian Uncertainty? | Architektur-spezifisch, hätte den TIMM-Generic-Loader gebrochen. |
| Warum keine Diversity-Sampling-Variante? | Erfordert paarweise Distanzen — O(N²) im Pool, würde die Live-UI ausbremsen. |
| Warum 30 Zyklen, nicht mehr? | Pro Zyklus ~20 min Training auf H100; 30 Zyklen zeigen Lernkurven-Plateau. |
| Was, wenn die GT-Labels nicht verfügbar wären? | Auto-Annotate ist nur für Experimente; UI hat schon manuellen Annotation-Modus mit Class-Picker. |
| Warum Streamlit und nicht Gradio/Dash? | Streamlit hat das beste Python-only-Erlebnis und @st.cache_resource für persistente Singletons. Limitierungen wurden bewusst akzeptiert und durch Threading-Modell adressiert. |
| Warum kein Vortraining mit SupCon allein, dann CE-Finetune? | Zweistufige Pipelines machen die UI komplexer und das Reset-Verhalten zwischen Zyklen unklar. Combined Loss ist einfacher und liefert ähnliche Ergebnisse. |
| Reproduzierbarkeit? | Seed 42 fixiert, Config wird pro Run als YAML gespeichert, alle Pool-Indizes geloggt. |

---

## Vortrags-Checkliste (Tag X)

- [ ] Stoppuhr-Test einmal durchgehen — Ziel 19:30–20:00.
- [ ] Live-Demo vorab mit `quick_test.yaml` proben (schneller Start).
- [ ] Backup-Screenshots für Demo-Steps falls Streamlit hängt.
- [ ] Wasser bereitstellen.
- [ ] Folie 5: "5 Tabs", nicht "4 Tabs" — auf der Folie steht "4", aber wir nennen alle 5.
- [ ] Folie 14-Zahlen (56% / 68% / ECE 0,16 vs. 0,19) gegen `experiments/*/al_cycle_results.json` prüfen.
