# Dataset & Execution Guide: Task A2 (Concrete Compressive Strength)

## 1. Dataset Overview
- **Dataset ID:** UCI Machine Learning Repository ID 165
- **Official URL:** [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Donor:** Prof. I-Cheng Yeh, Department of Civil Engineering, Chung-Hua University
- **Citation:** Yeh, I-Cheng. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12), 1797-1808.
- **Characteristics:** 1,030 instances, 8 numerical features, 1 continuous target (`compressive_strength` in MPa).
- **Domain Context:** Laboratory concrete testing requires up to 28 days for curing. Compressive strength development is non-linear (rapid rise in the first 7–14 days, plateauing toward 28 days). Abrams' Law dictates that the water-to-cement ratio ($w/c$) is inversely correlated with compressive strength.

---

## 2. Feature & Target Specification

| Feature Name | Clean Identifier | Physical Units | Description & Role |
| :--- | :--- | :--- | :--- |
| `Cement (component 1)` | `cement` | $\text{kg/m}^3$ | Primary binding agent |
| `Blast Furnace Slag (component 2)` | `slag` | $\text{kg/m}^3$ | Supplementary cementitious material (pozzolanic) |
| `Fly Ash (component 3)` | `fly_ash` | $\text{kg/m}^3$ | Secondary binder (improves long-term strength) |
| `Water (component 4)` | `water` | $\text{kg/m}^3$ | Activates hydration (excess causes porosity) |
| `Superplasticizer (component 5)` | `superplasticizer` | $\text{kg/m}^3$ | Chemical admixture (deflocculates cement, lowers water need) |
| `Coarse Aggregate (component 6)` | `coarse_aggregate` | $\text{kg/m}^3$ | Structural gravel/crushed stone filler |
| `Fine Aggregate (component 7)` | `fine_aggregate` | $\text{kg/m}^3$ | Sand filler |
| `Age` | `age` | Days ($1 \sim 365$) | Curing duration (heavily right-skewed) |
| **Concrete Compressive Strength** | `compressive_strength` | **MPa** | **Regression Target** |

### Engineered Features (Domain Additions)
1. **Water-to-Cement Ratio:** $w/c = \frac{\text{water}}{\text{cement}}$
2. **Logarithmic Age:** $\ln(\text{Age})$, linearizing hydration kinetics.

---

## 3. How to Execute `A2_notebook.ipynb` in Google Colab

To comply with Windows 11 Smart App Control policies without disabling security protections, all model training and evaluation experiments are designed to run seamlessly in **Google Colab**.

### Step 1: Open Google Colab
1. In your browser, open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `A2_Concrete_Strength/A2_notebook.ipynb` from this folder.

### Step 2: Run All Cells
1. In Google Colab, click **Runtime** in the top menu $\to$ **Run all** (or press `Ctrl + F9`).
2. The notebook is fully self-contained:
   - Cell 1 automatically installs `ucimlrepo`, `openpyxl`, and `xlrd`.
   - Cell 2 automatically downloads the official UCI Concrete dataset.
   - All EDA plots, feature engineering, 5-fold cross-validation tuning, ablation studies, and evaluation tables execute in **under 45 seconds**.

### Step 3: Save Executed Outputs Back to Project
1. **Download the executed notebook:**
   - In Colab: Click **File** $\to$ **Download** $\to$ **Download .ipynb**.
   - Save/replace it at `A2_Concrete_Strength/A2_notebook.ipynb` so all cell outputs, charts, and tables are preserved.
2. **Retrieve Figures and Metrics:**
   - The final cell in the notebook automatically creates and downloads `a2_artifacts.zip`.
   - Extract `a2_artifacts.zip` into `A2_Concrete_Strength/figures/`.
   - It contains:
     - `a2_eda_distributions.png`
     - `a2_correlation_heatmap.png`
     - `a2_eda_scatter_plots.png`
     - `a2_predicted_vs_actual.png`
     - `a2_feature_importance.png`
     - `a2_test_results.csv`
     - `a2_ablation_study.csv`
     - `a2_descriptive_statistics.csv`
