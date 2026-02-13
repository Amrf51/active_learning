# Visual and Interactive Active Learning:  
## A Streamlit-Based Framework for Investigating Sampling Strategies in Vehicle Image Classification with PyTorch

---

# 1. Project Title and Abstract (Short Summary)

## Project Title  
**Development of an Interactive Active Learning Pipeline for the Visual Analysis and Control of Training Processes in Vehicle Classification**

## Abstract  
In modern computer vision, the availability of high-quality annotated image data represents one of the greatest challenges. Especially in complex tasks such as vehicle classification, which can include both coarse categorization (car, truck, motorcycle) and fine-grained differentiation (e.g., by car brands and models), the manual annotation effort is enormous. Active Learning (AL) offers a promising solution by intelligently selecting those data points for annotation that are expected to provide the highest information gain for the model.

This thesis focuses on the design and implementation of a fully functional active learning pipeline. The core of the project is the development of a web-based GUI using Streamlit, which makes the entire AL cycle transparent and comprehensible. The pipeline, based on Python and PyTorch, enables users to compare different sampling strategies and neural network architectures. A particular emphasis is placed on the visual presentation of results, including “live training” feedback and a comprehensive evaluation pipeline to meet scientific standards. The goal is to develop a tool that not only demonstrates the efficiency of the labeling process but also provides deeper insights into the functionality and dynamics of active learning approaches.

---

# 2. Detailed Objective and Development Plan

This plan is divided into logical phases that are relevant both for the practical project (implementation) and for the bachelor thesis (scientific contextualization and evaluation).

---

# Phase 1: Fundamentals and Research

## Literature Review

- **Active Learning:** In-depth study of theoretical foundations. Focus on different sampling strategies (Query-by-Committee, Uncertainty Sampling, Diversity Sampling, etc.).
- **Neural Networks for Image Classification:** Analysis of state-of-the-art architectures (e.g., ResNet, MobileNet, EfficientNet) and their suitability for the given task.
- **Frameworks:** Familiarization with PyTorch for model development and Streamlit for GUI creation.

## Dataset Research and Selection

**Requirement:** A dataset that contains both coarse and fine vehicle classes. Ideally with a large number of unlabeled images.

**Recommendation:**

- **Stanford Cars Dataset:** Contains 16,185 images of 196 car brands and models. Suitable for fine-grained classification.
- **CompCars Dataset:** Comprehensive dataset with over 160,000 images, covering different viewpoints and a hierarchy of brands and models.

**Decision and Preprocessing:**  
Selection of a dataset. Splitting into an initial small labeled training set, an unlabeled pool, and a separate test set for final evaluation.

---

# Phase 2: System Architecture Design

## Definition of the Active Learning Pipeline

- **Initialization:** Train an initial model on a small, randomly selected labeled dataset.
- **Prediction:** Apply the current model to the pool of unlabeled data.
- **Querying/Sampling:** Apply a sampling strategy to select the most “informative” data points.
- **Annotation (simulated):** Add the selected data points (with their ground-truth labels) to the training dataset.
- **Re-Training:** Retrain the model with the expanded dataset.
- **Evaluation:** Measure model performance on a fixed test dataset.

## Design of the Streamlit GUI

### Dashboard Structure  
Design of a multi-page dashboard (e.g., via a sidebar).

### Components

- **Configuration Page:** Selection of neural network, sampling strategy, sampling batch size, etc.
- **Main Page (AL Cycle):** Visualization of the current state, display of selected images for annotation, button to start the next cycle.
- **Evaluation Page:** Graphical representation of performance metrics across AL cycles.
- **Dataset Explorer:** View for inspecting the labeled and unlabeled pools.

---

# Phase 3: Implementation of Core Functionality

## Backend (PyTorch & Python)

- Implementation of data loaders for labeled, unlabeled, and test datasets.
- Implementation of at least two different neural network architectures (e.g., a lightweight one like MobileNetV2 and a more powerful one like ResNet-50).
- Implementation of training and evaluation loops.
- Implementation of at least three different sampling strategies:
  - **Uncertainty Sampling** (e.g., Least Confidence): Simple and effective.
  - **Margin Sampling:** Measures the difference between the confidences of the two most probable classes.
  - **Entropy-based Sampling:** Considers uncertainty across all classes.

## Frontend (Streamlit)

- Development of the basic GUI structure.
- Integration of configuration options.
- Development of interactive control for the AL cycle (Start, Stop, Next Step).
- Display of selected images and (simulated) annotation interface.

---

# Phase 4: Integration and Live Training

- **Backend-Frontend Integration:** Ensure smooth communication. The pipeline state (current cycle, dataset sizes, etc.) must be reflected in the GUI.

## Implementation of “Live Training”

- Display training progress (loss curve, accuracy) in real time during the retraining phase. This can be implemented using Streamlit’s `st.empty()` and periodic updates.
- Visualization of model predictions on example images, updated after each cycle.

---

# Phase 5: Evaluation Pipeline and Visualization

## Implementation of Evaluation Metrics

- **Classification Metrics:** Accuracy, Precision, Recall, F1-Score (especially important for imbalanced classes).

## Visualization

- **Confusion Matrix:** For detailed analysis of misclassifications.
- **Performance Graphs:** Line charts showing metric development over the number of annotated samples (or AL cycles). This is the key graph to demonstrate the success of AL.
- **Comparison Graphs:** Enable direct comparison of performance curves across different sampling strategies or networks.

---

# Conducting Experiments

- Systematic testing of different configurations.
- Comparison of active learning strategies with a baseline (Random Sampling).
- Logging all results for inclusion in the bachelor thesis.
