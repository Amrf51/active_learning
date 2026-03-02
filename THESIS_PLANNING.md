# Thesis Planning: Research Questions & Experimental Setup

This document outlines potential research questions and a structured experimental plan for your Bachelor's thesis, based on the Streamlit Active Learning tool you have developed.

## 1. Defining the Core Thesis Question

Your thesis should revolve around a central research question (or a main question with sub-questions). Here are three distinct angles you can choose from, depending on what you want to focus on:

### Angle A: The Algorithmic Focus (Recommended as Primary)
**Main Question:** 
*"How do different uncertainty-based active learning strategies compare in terms of sample efficiency and classification performance when applied to fine-grained vehicle image classification?"*

**Sub-questions:**
- Does Active Learning significantly reduce the number of labeled images required to achieve a target accuracy compared to random sampling?
- Which specific querying strategy (Least Confidence, Margin, or Entropy) is most effective for distinguishing between visually similar vehicle classes?

### Angle B: The Systems & Human-in-the-Loop Focus
**Main Question:** 
*"To what extent can an interactive, visual active learning pipeline improve the transparency, evaluation, and manual control of the training process in computer vision tasks?"*

**Sub-questions:**
- How does real-time visual feedback (e.g., probe images, visual query presentation) assist in understanding model uncertainties?
- Can a decoupled architecture (UI vs. background worker) effectively handle the computationally intensive Active Learning cycles while maintaining a responsive user experience? *(You can use your `ARCHITECTURE_MAP.md` here!)*

### Angle C: The Architectural Impact Focus
**Main Question:** 
*"How does the architectural complexity of the baseline neural network interact with the effectiveness of active learning querying strategies?"*

**Sub-questions:**
- Do lightweight models (e.g., MobileNetV2) benefit more or less from intelligent sampling compared to high-capacity models (e.g., ResNet-50)?

---

## 2. Experimental Setup

To answer these questions, you can use your tool to run the following systematic experiments. Your tool's ability to save `al_cycle_results.json` and generate performance graphs is exactly what you need for this.

### Experiment 1: The Baseline Comparison (Essential)
**Goal:** Prove that Active Learning works and identify the best strategy.
- **Dataset:** Stanford Cars or CompCars (subset if necessary for time).
- **Model:** ResNet-50 (or a consistent baseline model).
- **Configurations to run (using Auto-Annotate for speed):**
  1. Random Sampling (Your baseline)
  2. Uncertainty Sampling (Least Confidence)
  3. Margin Sampling
  4. Entropy-based Sampling
- **Evaluation:** 
  - Plot a line graph of **Test Accuracy vs. Number of Labeled Samples** (or AL Cycles).
  - Expectation: The AL strategies should steepen the learning curve, meaning they reach (for example) 80% accuracy in fewer cycles than Random Sampling.

### Experiment 2: Resolving Fine-Grained Confusion (Deep Dive)
**Goal:** Show exactly *why* Active Learning is beneficial for difficult datasets like vehicles.
- **Method:** Look deeply into the results from Experiment 1.
- **Evaluation:**
  - Compare the **Confusion Matrices** of Random Sampling vs. Margin Sampling at the same cycle (e.g., Cycle 5).
  - Analyze the **Probe Images** (from your `state.py`). Show a visual example in your thesis of a confusing car image where the baseline model failed, but the Margin Sampling strategy queried similar images, leading the model to correct its prediction in later cycles.

### Experiment 3: Model Capacity Comparison (Optional but highly scientific)
**Goal:** Answer the question from Angle C.
- **Configurations:**
  - Model 1: MobileNetV2 (Lightweight)
  - Model 2: ResNet-50 (Heavy)
- **Runs per model:** 1x Random Sampling, 1x Best AL Strategy (e.g., Entropy).
- **Evaluation:** Does Active Learning help close the performance gap between a small model and a big model? 

---

## 3. How to Structure Your Written Thesis

1. **Introduction:** Introduce the problem of massive data requirements in Deep Learning and the manual annotation bottleneck, specifically for specialized fields like vehicle classification.
2. **Theoretical Background:**
   - Transfer Learning and CNN Basics (ResNet, MobileNet).
   - Active Learning paradigms (Pool-based sampling).
   - Math behind the specific strategies you implemented (Entropy, Margin, Least Confidence).
3. **Methodology & Tool Development (The Practical Part):**
   - Explain your system architecture (use `ARCHITECTURE_MAP.md`).
   - UI/UX design choices for the Human-in-the-Loop aspect.
   - The decoupled worker/controller approach you engineered.
4. **Experimental Setup:** Describe the dataset, the models, hyperparameters, and the structure of Experiments 1, 2, and 3.
5. **Results:** Present the graphs, confusion matrices, and visual probe image progressions generated by your tool.
6. **Discussion:** Interpret the results. Why did Margin sampling beat Least Confidence? What role did the visual interface play?
7. **Conclusion & Future Work:** Summarize findings and suggest next steps.
