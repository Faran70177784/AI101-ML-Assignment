# AI101 - Machine Learning Assignment: Three Data Modalities

**The University of Lahore**  
**Department of Intelligent Systems**  
**Course:** Introduction to Artificial Intelligence (AI-101) | Fall 2026  
**Instructor:** Dr. Muhammad Awais, AI Architect  

---

## Student Identification & Selected Tasks

| Field | Detail |
| :--- | :--- |
| **Student Name** | Syed Faran Ali |
| **Roll Number** | 866770 |
| **SAP ID** | 70177784 |
| **Section** | Section A |
| **GitHub Repository** | `https://github.com/Faran70177784/AI101-ML-Assignment` |
| **Graded Pool A Task** | **A2**: Predicting Concrete Compressive Strength (Numerical Regression) |
| **Graded Pool B Task** | **B3**: Predicting Star Ratings of Clothing Reviews (Mixed Tabular + Text) |
| **Graded Pool C Task** | **C3**: Image Captioning with CNN-LSTM (Seq2Seq Image-to-Text) |
| **Bonus Tasks** | None (Attempting 3 required graded tasks) |

---

## Repository Structure & Deliverables

```text
AI101-ML-Assignment/
│
├── README.md                          <- Project overview, student info, and run guide
├── PROJECT_PLAN.md                    <- 8-phase engineering and delivery plan
├── REQUIREMENTS_TRACKER.md            <- Complete mapping of mandatory requirements & verification status
├── AI_DISCLOSURE.md                   <- Official AI use disclosure statement per assignment rules
├── requirements.txt                   <- Complete Python dependencies list
├── .gitignore                         <- Excludes datasets, venv, and large model weights (>50MB)
│
├── docs/                              <- Official assignment and template PDFs
│   ├── Assignment_ML_Three_Data_Modalities.pdf
│   └── Report_Template_Two_Column.pdf
│
├── A2_Concrete_Strength/              <- Pool A: Numerical Regression
│   ├── A2_notebook.ipynb              <- Executed Jupyter notebook with all outputs
│   ├── data/
│   │   └── README.md                  <- Dataset acquisition guide (UCI ID 165)
│   ├── figures/                       <- Generated EDA, residual, and regression plots
│   ├── models/                        <- Saved model parameters / pipelines (<50MB)
│   └── report/
│       ├── 70177784_A2_Report.pdf     <- Final 2-page double-column IEEE report
│       └── source/
│           └── A2_Report.tex          <- Complete LaTeX source
│
├── B3_Clothing_Reviews/               <- Pool B: Mixed Modality Classification
│   ├── B3_notebook.ipynb              <- Executed Jupyter notebook with all outputs
│   ├── data/
│   │   └── README.md                  <- Dataset acquisition guide (Kaggle)
│   ├── figures/                       <- Generated rating distributions, confusion matrices
│   ├── models/                        <- Saved model checkpoints (<50MB)
│   └── report/
│       ├── 70177784_B3_Report.pdf     <- Final 2-page double-column IEEE report
│       └── source/
│           └── B3_Report.tex          <- Complete LaTeX source
│
├── C3_Image_Captioning/               <- Pool C: Unstructured Seq2Seq
│   ├── C3_notebook.ipynb              <- Executed Jupyter notebook with all outputs
│   ├── data/
│   │   └── README.md                  <- Dataset acquisition guide (Flickr8k)
│   ├── figures/                       <- Training loss curves, qualitative sample captions
│   ├── models/                        <- Checkpoints & vocabulary artifacts (<50MB)
│   └── report/
│       ├── 70177784_C3_Report.pdf     <- Final 2-page double-column IEEE report
│       └── source/
│           └── C3_Report.tex          <- Complete LaTeX source
│
└── scripts/
    └── environment_check.py           <- Automated hardware & Python dependency verifier
```

---

## Quick Links to Notebooks & Reports

| Task ID | Task Description | Executed Notebook | Final Report (PDF) | Report Source |
| :--- | :--- | :--- | :--- | :--- |
| **A2** | Concrete Compressive Strength | [A2_notebook.ipynb](A2_Concrete_Strength/A2_notebook.ipynb) | [70177784_A2_Report.pdf](A2_Concrete_Strength/report/70177784_A2_Report.pdf) | [A2_Report.tex](A2_Concrete_Strength/report/source/A2_Report.tex) |
| **B3** | Clothing Review Star Ratings | [B3_notebook.ipynb](B3_Clothing_Reviews/B3_notebook.ipynb) | [70177784_B3_Report.pdf](B3_Clothing_Reviews/report/70177784_B3_Report.pdf) | [B3_Report.tex](B3_Clothing_Reviews/report/source/B3_Report.tex) |
| **C3** | Flickr8k Image Captioning | [C3_notebook.ipynb](C3_Image_Captioning/C3_notebook.ipynb) | [70177784_C3_Report.pdf](C3_Image_Captioning/report/70177784_C3_Report.pdf) | [C3_Report.tex](C3_Image_Captioning/report/source/C3_Report.tex) |

---

## Environment Setup & Execution Guide

### 1. Prerequisites
- Python 3.11 (Official course standard)
- Recommended: 8+ GB RAM, CPU or optional CUDA GPU

### 2. Virtual Environment Setup
```bash
# Verify Python 3.11 is available
py -3.11 --version

# Create dedicated virtual environment
py -3.11 -m venv ai-lab

# Activate virtual environment
# On Windows (PowerShell):
.\ai-lab\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\ai-lab\Scripts\activate.bat
```

### 3. Install Required Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Environment Diagnostic
```bash
python scripts/environment_check.py
```

### 5. Running the Notebooks
Launch Jupyter Lab or Notebook server:
```bash
jupyter notebook
```
Navigate to the respective task directory and execute cells sequentially from top to bottom.
