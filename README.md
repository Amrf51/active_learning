

### **Titelvorschlag**


* **Visuelles und interaktives Active Learning: Ein Streamlit-basiertes Framework zur Untersuchung von Sampling-Strategien in der Fahrzeugbildklassifikation mit PyTorch**

---

### **1. Projekttitel und Abstract (Kurzzusammenfassung)**

**Projekttitel:** Entwicklung einer interaktiven Active-Learning-Pipeline zur visuellen Analyse und Steuerung von Trainingsprozessen für die Fahrzeugklassifikation

**Abstract:** In der modernen Computer Vision stellt die Verfügbarkeit von qualitativ hochwertig annotierten Bilddaten eine der größten Herausforderungen dar. Insbesondere bei komplexen Aufgaben wie der Fahrzeugklassifikation, die sowohl eine grobe Einteilung (PKW, LKW, Motorrad) als auch eine feingranulare Differenzierung (z.B. nach Automarken und -modellen) umfassen kann, ist der manuelle Annotationsaufwand enorm. Das Active Learning (AL) verspricht hier Abhilfe, indem es intelligent diejenigen Datenpunkte zur Annotation auswählt, die den größten Informationsgewinn für das Modell versprechen.

Diese Arbeit befasst sich mit der Konzeption und Implementierung einer voll funktionsfähigen Active-Learning-Pipeline. Kern des Projekts ist die Entwicklung einer webbasierten GUI mittels Streamlit, die den gesamten AL-Zyklus transparent und nachvollziehbar macht. Die Pipeline, basierend auf Python und PyTorch, ermöglicht es dem Nutzer, verschiedene Sampling-Strategien und neuronale Netzarchitekturen zu vergleichen. Ein besonderer Fokus liegt auf der visuellen Aufbereitung der Ergebnisse, inklusive "Live-Training"-Feedback und einer umfassenden Evaluations-Pipeline, um den wissenschaftlichen Ansprüchen gerecht zu werden. Das Ziel ist ein Werkzeug, das nicht nur die Effizienz des Labeling-Prozesses demonstriert, sondern auch tiefere Einblicke in die Funktionsweise und Dynamik von Active-Learning-Ansätzen ermöglicht.

---

### **2. Detaillierter Ziel- und Entwicklungsplan**

Dieser Plan ist in logische Phasen unterteilt, die sowohl für das Praxisprojekt (Implementierung) als auch für die Bachelorarbeit (wissenschaftliche Einordnung und Evaluation) relevant sind.

#### **Phase 1: Grundlagen und Recherche **

* **Literaturrecherche:**
    * **Active Learning:** Vertiefende Einarbeitung in die theoretischen Grundlagen. Fokus auf unterschiedliche Sampling-Strategien (Query-by-Committee, Uncertainty Sampling, Diversity Sampling, etc.).
    * **Neuronale Netze für Bildklassifikation:** Analyse von State-of-the-Art-Architekturen (z.B. ResNet, MobileNet, EfficientNet) und deren Eignung für die gegebene Aufgabenstellung.
    * **Frameworks:** Einarbeitung in PyTorch für die Modellentwicklung und Streamlit für die GUI-Erstellung.
* **Datensatz-Recherche und -Auswahl:**
    * **Anforderung:** Ein Datensatz, der sowohl grobe als auch feine Fahrzeugklassen enthält. Idealerweise mit einer großen Anzahl ungelabelter Bilder.
    * **Empfehlung:**
        * **Stanford Cars Dataset:** Enthält 16.185 Bilder von 196 Automarken und -modellen. Gut für die Feinklassifikation.
        * **CompCars Dataset:** Umfassender Datensatz mit über 160.000 Bildern, der verschiedene Blickwinkel und eine Hierarchie von Marken und Modellen bietet.
    * **Entscheidung und Vorverarbeitung:** Auswahl eines Datensatzes. Aufteilung in einen initialen, kleinen gelabelten Trainingsdatensatz, einen ungelabelten Pool und ein separates Testset zur finalen Evaluation.

#### **Phase 2: Konzeption der Systemarchitektur **

* **Definition der Active-Learning-Pipeline:**
    1.  **Initialisierung:** Training eines initialen Modells auf einem kleinen, zufällig ausgewählten, gelabelten Datensatz.
    2.  **Prädiktion:** Anwendung des aktuellen Modells auf den Pool ungelabelter Daten.
    3.  **Querying/Sampling:** Anwendung einer Sampling-Strategie, um die "informativsten" Datenpunkte auszuwählen.
    4.  **Annotation (simuliert):** Hinzufügen der ausgewählten Datenpunkte (mit ihren Ground-Truth-Labels) zum Trainingsdatensatz.
    5.  **Re-Training:** Neutraining des Modells mit dem erweiterten Datensatz.
    6.  **Evaluation:** Messung der Modellperformance auf einem festen Testdatensatz.
* **Design der Streamlit-GUI:**
    * **Dashboard-Struktur:** Entwurf eines mehrseitigen Dashboards (z.B. über eine Seitenleiste).
    * **Komponenten:**
        * **Konfigurationsseite:** Auswahl des neuronalen Netzes, der Sampling-Strategie, der Batch-Größe für das Sampling, etc.
        * **Hauptseite (AL-Zyklus):** Visualisierung des aktuellen Zustands, Anzeige der zur Annotation ausgewählten Bilder, Button zum Starten des nächsten Zyklus.
        * **Evaluationsseite:** Grafische Darstellung der Performance-Metriken über die AL-Zyklen.
        * **Datensatz-Explorer:** Eine Ansicht zur Inspektion des gelabelten und ungelabelten Pools.

#### **Phase 3: Implementierung der Kernfunktionalität **

* **Backend (PyTorch & Python):**
    * Implementierung der Datenlader für den gelabelten, ungelabelten und Test-Datensatz.
    * Implementierung von mindestens zwei verschiedenen neuronalen Netz-Architekturen (z.B. ein schlankes wie MobileNetV2 und ein leistungsstärkeres wie ResNet-50).
    * Implementierung des Trainings- und Evaluations-Loops.
    * Implementierung von mindestens drei verschiedenen Sampling-Strategien:
        1.  **Uncertainty Sampling (z.B. Least Confidence):** Einfach und effektiv.
        2.  **Margin Sampling:** Misst die Differenz zwischen den Konfidenzen der beiden wahrscheinlichsten Klassen.
        3.  **Entropy-based Sampling:** Berücksichtigt die Unsicherheit über alle Klassen.
* **Frontend (Streamlit):**
    * Aufbau der grundlegenden GUI-Struktur.
    * Integration der Konfigurationsmöglichkeiten.
    * Entwicklung der interaktiven Steuerung für den AL-Zyklus (Start, Stopp, Nächster Schritt).
    * Anzeige der ausgewählten Bilder und (simulierten) Annotations-Schnittstelle.

#### **Phase 4: Integration und Live-Training **

* **Verbindung von Backend und Frontend:** Sicherstellung einer reibungslosen Kommunikation. Der Zustand der Pipeline (aktueller Zyklus, Datensatzgrößen, etc.) muss in der GUI reflektiert werden.
* **Implementierung des "Live-Trainings":**
    * Anzeige des Trainingsfortschritts (Loss-Kurve, Genauigkeit) in Echtzeit während der Re-Training-Phase. Dies kann über Streamlits `st.empty()` und regelmäßige Updates realisiert werden.
    * Visualisierung der Vorhersagen des Modells auf Beispielbildern, die sich nach jedem Zyklus aktualisieren.

#### **Phase 5: Evaluationspipeline und Visualisierung **

* **Implementierung der Evaluationsmetriken:**
    * **Klassifikationsmetriken:** Genauigkeit (Accuracy), Präzision, Recall, F1-Score (insbesondere bei unbalancierten Klassen wichtig).
    * **Visualisierung:**
        * **Confusion Matrix:** Zur detaillierten Analyse von Fehlklassifikationen.
        * **Performance-Graphen:** Liniendiagramme, die die Entwicklung der Metriken über die Anzahl der annotierten Samples (oder AL-Zyklen) zeigen. Dies ist der Kerngraph, um den Erfolg von AL zu beweisen.
        * **Vergleichs-Graphen:** Ermöglichen den direkten Vergleich der Performance-Kurven verschiedener Sampling-Strategien oder Netzwerke.
* **Durchführung der Experimente:**
    * Systematisches Testen der verschiedenen Konfigurationen.
    * Vergleich der Active-Learning-Strategien mit einer Baseline (Random Sampling).
    * Protokollierung aller Ergebnisse für die Bachelorarbeit.


