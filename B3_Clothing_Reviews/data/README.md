# Dataset & Execution Guide: Task B3 (Clothing Review Star Ratings)

## 1. Dataset Overview
- **Dataset Source:** Kaggle: `nicapotato/womens-ecommerce-clothing-reviews`
- **URL:** [Women's E-Commerce Clothing Reviews on Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- **Author:** Nicholasmcdonald (Kaggle)
- **License:** CC0: Public Domain
- **Size & Shape:** 23,486 reviews $\times$ 10 columns; target: Rating 1–5 (heavily skewed towards 5 stars).
- **Cleaned Size:** 22,641 reviews after dropping rows with empty `Review Text` (845 rows removed).

---

## 2. Feature Specification & Strict Leakage Rule

| Feature Name | Clean Identifier | Type | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| `Review Text` | `full_text` | Text | Concatenated with Title; processed via TF-IDF (15,000 n-grams) |
| `Title` | `Title` | Text | Review title merged into full text |
| `Age` | `Age` | Numerical | Standardized with `StandardScaler` |
| `Positive Feedback Count` | `log_feedback` | Numerical | Log-transformed $\ln(1 + x)$ and standardized |
| `char_length`, `word_count` | `word_count` | Numerical | Word count of merged text |
| `Division Name` | `Division Name` | Categorical | One-Hot Encoded (General, General Petite, Initmates) |
| `Department Name` | `Department Name` | Categorical | One-Hot Encoded (Dresses, Tops, Bottoms, etc.) |
| `Class Name` | `Class Name` | Categorical | One-Hot Encoded (Pants, Blouses, Knits, etc.) |
| **Rating** | `Rating` | **Ordinal (1--5)** | **Target Variable** |

> ### ⚠️ Mandatory Data Leakage Rule
> **Under NO circumstances may `Recommended IND` be used as an input feature.**  
> *Reasoning:* The customer marks their recommendation indicator simultaneously when selecting their star rating. In real-world deployment, this column is not available beforehand; incorporating it introduces fatal label leakage and artificially inflates accuracy without real-world utility.

---

## 3. How to Execute `B3_notebook.ipynb` in Google Colab

To comply with Windows 11 Smart App Control policies without disabling security protections, all model training and evaluation experiments are designed to run seamlessly in **Google Colab**.

### Step 1: Open Google Colab
1. In your browser, open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `B3_Clothing_Reviews/B3_notebook.ipynb` from this folder.

### Step 2: Run All Cells
1. In Google Colab, click **Runtime** in the top menu $\to$ **Run all** (or press `Ctrl + F9`).
2. The notebook is fully self-contained:
   - Cell 1 automatically verifies dependencies.
   - Cell 2 automatically fetches the official Women's Clothing Reviews dataset from its direct mirror.
   - All EDA plots, text preprocessing, multimodal `ColumnTransformer` pipelines, class-weighting comparisons, modality ablation experiments, 3-class reframing, and confusion matrices execute in **under 90 seconds**.

### Step 3: Save Executed Outputs Back to Project
1. **Download the executed notebook:**
   - In Colab: Click **File** $\to$ **Download** $\to$ **Download .ipynb**.
   - Save/replace it at `B3_Clothing_Reviews/B3_notebook.ipynb` so all cell outputs, charts, and tables are preserved.
2. **Retrieve Figures and Metrics:**
   - The final cell in the notebook automatically creates and downloads `b3_artifacts.zip`.
   - Extract `b3_artifacts.zip` into `B3_Clothing_Reviews/figures/`.
   - It contains:
     - `b3_rating_distribution.png`
     - `b3_review_length_and_age.png`
     - `b3_ratings_by_department.png`
     - `b3_confusion_matrix.png`
     - `b3_class_weighting_comparison.csv`
     - `b3_ablation_study.csv`
     - `b3_3class_reframing_results.csv`
     - `b3_error_analysis_samples.csv`
