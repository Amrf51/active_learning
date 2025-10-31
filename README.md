# 🚗 Active Learning for Car Classification  
### Streamlit-based Visual Framework using PyTorch

**Author:** Amro Elsaadany  
**University:** FH Aachen – University of Applied Sciences  
**Thesis Title:** *Visuelles und interaktives Active Learning: Ein Streamlit-basiertes Framework zur Untersuchung von Sampling-Strategien in der Fahrzeugbildklassifikation*  

---

## 🧠 Overview

This project implements an **interactive Active Learning (AL) framework** for **vehicle image classification**, combining **PyTorch** for model training and **Streamlit** for visual control and monitoring.

The goal is to make the **Active Learning cycle** transparent and explorable — showing how different sampling strategies impact model performance over time.

---

## 🎯 Objectives

- Develop a **modular AL pipeline** for iterative model improvement  
- Compare **multiple sampling strategies** (Uncertainty, Margin, Entropy)  
- Integrate **real-time visual feedback** (training metrics, confusion matrices, etc.)  
- Provide an **interactive Streamlit dashboard** to control and visualize each AL cycle  

---

## 🧩 System Architecture

```text
+---------------------------+
|   Dataset (Stanford Cars) |
+-------------+-------------+
              |
              v
+-------------+-------------+
| Active Learning Pipeline  |
| - Data Loader (labeled/unlabeled/test) |
| - Model (ResNet, MobileNet)            |
| - Sampler (Uncertainty, Margin, Entropy)|
+-------------+-------------+
              |
              v
+---------------------------+
|   Streamlit Dashboard     |
| - Configurations           |
| - Cycle Control (Next, Stop)|
| - Performance Visuals      |
+---------------------------+
