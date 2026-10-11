# Artificial Neural Networks (ANN) - Assignment No 2

## Multi-Layer Neural Network (MLNN / MLP) - Student Exam Score Predictor
**Instructor:** Mam Saima Hassan  
**Course:** Artificial Neural Networks (ANN) - BS Computer Science (7th Semester)  

---

## 📌 Project Overview
This repository contains the complete implementation of a **Multi-Layer Neural Network (MLNN / MLP)** for regression to predict student **Final Exam Scores** based on three input features:
1. **Study Hours** ($x_1$)
2. **Attendance Percentage** ($x_2$)
3. **Previous Test Score** ($x_3$)

The project includes:
- **`Multi_Layer_Neural_Network.xlsx`**: An executive, interactive Excel workbook with dynamic formulas for forward pass, live prediction form, backpropagation walkthrough, and Excel Solver training setup.
- **`train_mlnn_excel.py`**: Standalone Python script that trains the MLNN model, evaluates metrics, and generates the Excel workbook.
- **`Multi_Layer_Neural_Network.ipynb`**: Complete Jupyter Notebook for interactive execution and visualization.
- **`mlnn_model_evaluation.csv` & `mlnn_predictions.csv`**: Evaluation metrics and model predictions.

---

## 🧠 Neural Network Architecture

- **Input Layer:** 3 Nodes (`Study_Hours`, `Attendance_Percent`, `Previous_Test_Score`)
- **Hidden Layer:** 3 Neurons with Logistic (**Sigmoid**) activation function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Output Layer:** 1 Neuron for continuous regression prediction (`Final_Exam_Score`)
- **Feature Scaling:** Min-Max Normalization to $[0, 1]$

---

## 📊 Performance Benchmark

| Model | Architecture | Test MAE | Test RMSE | Test $R^2$ | Status |
|---|---|:---:|:---:|:---:|:---:|
| **Multiple Linear Regression (Assignment 1)** | Ordinary Least Squares | 0.76 | 0.95 | 0.9950 | Baseline |
| **Multi-Layer Neural Network (Assignment 2)** | **3-3-1 MLP (Sigmoid Hidden)** | **0.68** | **0.73** | **0.9971** | **Superior** |

---

## 📁 Repository Structure
```text
├── student_exam_scores.csv          # Raw dataset (20 student samples)
├── Multi_Layer_Neural_Network.xlsx  # Complete interactive Excel workbook with formulas
├── Multi_Layer_Neural_Network.ipynb # Jupyter Notebook for GitHub rendering
├── train_mlnn_excel.py              # Python training & Excel generator script
├── mlnn_model_evaluation.csv        # Performance evaluation metrics
├── mlnn_predictions.csv             # Full model predictions and residuals
├── model_evaluation.csv             # Baseline model metrics (Assignment 1)
└── README.md                        # Documentation
```

---

## 🚀 How to Run

### 1. Run Python Training & Excel Generator
```powershell
python train_mlnn_excel.py
```

### 2. Run in Excel (Using Excel Solver)
1. Open `Multi_Layer_Neural_Network.xlsx`.
2. Go to sheet **`Interactive Predictor`** to test custom student inputs in real-time.
3. To train/optimize inside Excel:
   - Go to **`Forward Pass (All Data)`** sheet.
   - Click **Data** tab -> **Solver**.
   - Target Cell: `$S$28` (Total SSE) -> Set to **Min**.
   - Changing Cells: Weights in **`Weights & Biases`** sheet.
   - Solving Engine: **GRG Nonlinear**.
   - Click **Solve**.
