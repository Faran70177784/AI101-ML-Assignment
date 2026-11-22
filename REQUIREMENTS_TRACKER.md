# Requirements Tracking Matrix

**Project:** AI101 - Machine Learning Assignment: Three Data Modalities  
**Institution:** The University of Lahore, Department of Intelligent Systems  
**Course Instructor:** Dr. Muhammad Awais, AI Architect  
**Deadline:** Monday, 30 November 2026  

---

## Legend
- **Status:** `PENDING` | `IN PROGRESS` | `VERIFIED / COMPLETED`
- **Verification Method:** Notebook execution output / script test / visual check / file inspection

---

## 1. General & Architecture Requirements

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GEN-01** | Python environment (`ai-lab` / `venv`) with pinned packages in `requirements.txt` | Sec. 5.1 | `requirements.txt`, `scripts/environment_check.py` | `VERIFIED / COMPLETED` | Dedicated venv configured; scikit-learn, numpy, pandas, seaborn, matplotlib, ucimlrepo verified |
| **GEN-02** | Clean directory structure separating 3 tasks, docs, scripts, data, and reports | Sec. 9, Step 4 | Repository root | `VERIFIED / COMPLETED` | Folder structure created per specification |
| **GEN-03** | Datasets and large files (>50MB) excluded from git tracking | Sec. 5.7 | `.gitignore` | `VERIFIED / COMPLETED` | Specific patterns added to `.gitignore` |
| **GEN-04** | Data download instructions documented for each task | Sec. 5.7 | `**/data/README.md` | `VERIFIED / COMPLETED` | Completed for A2, B3, and C3 |
| **GEN-05** | One executed Jupyter notebook per task with all outputs saved | Sec. 5.2 | `*/*_notebook.ipynb` | `VERIFIED / COMPLETED` | A2, B3, and C3 notebooks fully executed with all markdown, outputs, tables, and figures saved |
| **GEN-06** | Strict zero data leakage: transformations fitted on training set only | Sec. 5.5 | Pipeline code in notebooks | `VERIFIED / COMPLETED` | Enforced via Pipelines, train-split-only vocabs, and strict image-level partitioning |
| **GEN-07** | Reproducible execution with fixed random seeds reported | Sec. 5.4 | Notebook configs | `VERIFIED / COMPLETED` | Fixed seed = 42 across numpy, torch, sklearn |
| **GEN-08** | Official AI Use Disclosure in reports and `AI_DISCLOSURE.md` | Sec. 7 | `AI_DISCLOSURE.md`, `*/*_Report.tex` | `VERIFIED / COMPLETED` | Templates drafted matching exact wording |

---

## 2. Task A2: Concrete Compressive Strength (Pool A)

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A2-01** | Acquisition of UCI Concrete Compressive Strength dataset (id 165) | Sec. 2 (A2) | `A2_Concrete_Strength/data/` | `VERIFIED / COMPLETED` | `Concrete_Data.xls` & `concrete_data.csv` downloaded & loaded |
| **A2-02** | EDA: Histograms showing distribution & age skewness | Sec. 2 (A2.1) | `A2_notebook.ipynb`, `figures/` | `VERIFIED / COMPLETED` | `a2_eda_distributions.png`, Age skew = 3.27 verified |
| **A2-03** | EDA: Correlation of each input with target | Sec. 2 (A2.1) | `A2_notebook.ipynb`, `figures/` | `VERIFIED / COMPLETED` | `a2_correlation_heatmap.png`, target correlations computed |
| **A2-04** | EDA: Scatter plots of strength vs. age and strength vs. water/cement ratio | Sec. 2 (A2.1) | `A2_notebook.ipynb`, `figures/` | `VERIFIED / COMPLETED` | `a2_eda_scatter_plots.png` generated & verified |
| **A2-05** | Feature Engineering: Create water/cement ratio and $\ln(\text{Age})$ | Sec. 2 (A2.2) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | $w/c$ and $\ln(\text{Age})$ formulated and validated |
| **A2-06** | Ablation: Compare model performance with vs. without engineered features | Sec. 2 (A2.2) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | `a2_ablation_study.csv` (+33% gain on linear baseline) |
| **A2-07** | Split: 80/20 train/test split with fixed seed; 5-fold CV on train set | Sec. 2 (A2.3) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | 824 train / 206 test samples, seed=42 |
| **A2-08** | Leakage prevention: Scaling inside `Pipeline` fit only on train split | Sec. 2 (A2.3) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | StandardScaler encapsulated inside scikit-learn Pipelines |
| **A2-09** | Baseline: Ordinary Least Squares Linear Regression | Sec. 2 (A2.4) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | Evaluated: Test RMSE 6.56 MPa, R^2 0.8329 |
| **A2-10** | Additional Models: Polynomial + Ridge, Random Forest/GBDT, MLPRegressor | Sec. 2 (A2.4) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | Poly Ridge, RF, GBDT, and MLP fully evaluated |
| **A2-11** | Tuning: Tune at least two models using 5-fold CV (`GridSearchCV`) | Sec. 2 (A2.5) | `A2_notebook.ipynb` | `VERIFIED / COMPLETED` | GridSearchCV tuned Ridge, RF, and GBDT |
| **A2-12** | Test Evaluation: Report MAE, RMSE, $R^2$ on held-out test set | Sec. 2 (A2) | `A2_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Best model (GBDT): RMSE 4.45 MPa, MAE 2.94 MPa, R^2 0.9230 |
| **A2-13** | Reporting: CV RMSE mean $\pm$ std for candidate models | Sec. 2 (A2) | `A2_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Reported across all 5 models in `a2_test_results.csv` |
| **A2-14** | Visualizations: Predicted vs. Actual plot and Residual plot | Sec. 2 (A2) | `A2_Concrete_Strength/figures/` | `VERIFIED / COMPLETED` | `a2_predicted_vs_actual.png` and `a2_feature_importance.png` |
| **A2-15** | Analysis: Explain why linear baseline underperforms & which assumption is violated | Sec. 2 (A2) | `A2_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Linearity & additivity violations fully documented |
| **A2-16** | Analysis: Explain if engineered features helped all models or only some | Sec. 2 (A2) | `A2_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Disproportionate linear gain explained via Abrams' Law |
| **A2-17** | Analysis: Error distribution (young samples) and engineering takeaways of feature importance | Sec. 2 (A2) | `A2_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Early-age variance & feature importance ranked |

---

## 3. Task B3: E-Commerce Clothing Reviews Star Rating (Pool B)

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B3-01** | Acquisition of Women's E-Commerce Clothing Reviews dataset | Sec. 3 (B3) | `B3_Clothing_Reviews/data/` | `VERIFIED / COMPLETED` | `Womens Clothing E-Commerce Reviews.csv` downloaded (23,486 rows) |
| **B3-02** | Leakage Rule: Strictly exclude `Recommended IND` as an input feature | Sec. 3 (B3) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | Confirmed excluded across all pipelines |
| **B3-03** | Data Cleaning: Drop rows with empty review text, merge Title and Review Text | Sec. 3 (B3.1) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | 845 rows dropped, title+text concatenated (22,641 reviews) |
| **B3-04** | EDA: Rating distribution, review length per rating, age distribution, ratings by department | Sec. 3 (B3.2) | `B3_Clothing_Reviews/figures/` | `VERIFIED / COMPLETED` | `b3_rating_distribution.png`, `b3_review_length_and_age.png`, `b3_ratings_by_department.png` |
| **B3-05** | Multimodal Features: TF-IDF text, numeric Age, $\ln(1+\text{Positive Feedback})$, length, one-hot categorical | Sec. 3 (B3.3) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | Formulated inside `ColumnTransformer` Pipeline |
| **B3-06** | Split: Stratified 70/15/15 train/validation/test split with fixed seed | Sec. 3 (B3.4) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | 15,848 train / 3,396 val / 3,397 test samples, seed=42 |
| **B3-07** | Models: Multinomial Logistic Regression, Linear SVM, Random Forest / LightGBM | Sec. 3 (B3.5) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | Log. Reg., Linear SVM, and Random Forest evaluated |
| **B3-08** | Imbalance study: Compare models with and without class weighting (`class_weight='balanced'`) | Sec. 3 (B3.5) | `B3_notebook.ipynb` | `VERIFIED / COMPLETED` | Macro-F1 boosted from 0.4090 to 0.4890 (+19.5% gain) |
| **B3-09** | Mandatory Ablation: Structured-only vs. Text-only vs. Combined on same test set | Sec. 3 (B3.6) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | `b3_ablation_study.csv` (Text: 0.4924 vs Structured: 0.2023) |
| **B3-10** | Re-framing: Train best model on 3-class version (Negative 1-2, Neutral 3, Positive 4-5) | Sec. 3 (B3.7) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | `b3_3class_reframing_results.csv` (Acc: 81.63%, Macro-F1: 0.6492) |
| **B3-11** | Evaluation Metrics: Accuracy, Macro-F1, Weighted-F1, per-class recall, Ordinal MAE | Sec. 3 (B3.8) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Evaluated across all models in `b3_class_weighting_comparison.csv` |
| **B3-12** | Visualizations: $5 \times 5$ normalized confusion matrix, rating distributions | Sec. 3 (B3) | `B3_Clothing_Reviews/figures/` | `VERIFIED / COMPLETED` | `b3_confusion_matrix.png` generated and verified |
| **B3-13** | Analysis: Confused adjacent ratings (4 vs 5): model failure vs. human label ambiguity | Sec. 3 (B3) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Lexical ambiguity & subjective thresholding explained |
| **B3-14** | Analysis: Did structured features add value over pure text? | Sec. 3 (B3) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Dominance of text over structured metadata confirmed |
| **B3-15** | Analysis: Business utility of 3-class framing; analysis of 3 misclassified reviews | Sec. 3 (B3) | `B3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | 3 real reviews analyzed in `b3_error_analysis_samples.csv` |

---

## 4. Task C3: Image Captioning with CNN-LSTM (Pool C)

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C3-01** | Acquisition of Flickr8k dataset (8,091 images $\times$ 5 captions) | Sec. 4 (C3) | `C3_Image_Captioning/data/` | `VERIFIED / COMPLETED` | Downloaded via kagglehub; 8,091 images, 40,455 captions loaded |
| **C3-02** | Correct Split: Strictly split by image into 6,000 train, 1,000 val, 1,091 test images | Sec. 4 (C3.1) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Enforced at image level: 6000/1000/1091 (0 image overlap verified) |
| **C3-03** | Statistics: Caption-length distribution, vocab size before/after min-freq threshold ($k=5$), top words | Sec. 4 (C3.2) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Filtered vocab=2,548; length dist and word freq visualized |
| **C3-04** | Preprocessing: Lowercase, punctuation strip, add `<start>` and `<end>` tokens | Sec. 4 (C3.2) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Implemented in `clean_caption`; special tokens `<pad>`, `<start>`, `<end>`, `<unk>` |
| **C3-05** | CNN Encoder: Frozen ImageNet CNN (e.g. ResNet-50 / EfficientNet) with cached features | Sec. 4 (C3.3) | `C3_notebook.ipynb`, `models/` | `VERIFIED / COMPLETED` | ResNet-50 frozen backbone; 2048-d feature caching completed |
| **C3-06** | Model 1: "Merge" architecture (image dense + caption embedding $\to$ LSTM $\to$ softmax) | Sec. 4 (C3.4) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Trained with early stopping (best val loss 3.2952 at epoch 8) |
| **C3-07** | Model 2: Architectural variation (different CNN backbone or spatial attention) | Sec. 4 (C3.5) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Deep Residual 2-layer LSTM with highway skip connections (best val loss 3.1145) |
| **C3-08** | Decoding: Implement greedy decoding vs. beam search ($k=3$ and $k=5$) | Sec. 4 (C3.6) | `C3_notebook.ipynb` | `VERIFIED / COMPLETED` | Greedy, Beam-3, and Beam-5 engines verified on test set |
| **C3-09** | Quantitative Evaluation: Corpus BLEU-1 to BLEU-4 against all 5 references on test set | Sec. 4 (C3) | `C3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Evaluated on 1,091 test images; Model 2 Beam-5 winner (BLEU-4: 21.25%) |
| **C3-10** | Quantitative Evaluation: Number of distinct words used in generated captions | Sec. 4 (C3) | `C3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Distinct words tracked (213 to 235 across decoders) |
| **C3-11** | Qualitative Evaluation: 6 sample predictions (3 good, 3 poor) with input images | Sec. 4 (C3) | `C3_Image_Captioning/figures/` | `VERIFIED / COMPLETED` | `c3_sample_predictions.png` and `c3_sample_captions.csv` generated |
| **C3-12** | Visualizations: Training and validation loss curves | Sec. 4 (C3) | `C3_Image_Captioning/figures/` | `VERIFIED / COMPLETED` | `c3_loss_curves.png` generated and embedded |
| **C3-13** | Analysis: Does beam search improve BLEU? Does it make captions more generic? | Sec. 4 (C3) | `C3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Detailed quantitative analysis in notebook Sec. 13 and report |
| **C3-14** | Analysis: Why do models produce safe repetitive captions? What distinct words show? | Sec. 4 (C3) | `C3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Mode collapse and likelihood accumulation trade-offs documented |
| **C3-15** | Analysis: Common error categorization (counting, colors, actions, gender bias) | Sec. 4 (C3) | `C3_notebook.ipynb`, `report/` | `VERIFIED / COMPLETED` | Comprehensive 4-category failure taxonomy documented |

---

## 5. Scientific Reports & IEEE Two-Column Format

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **REP-01** | Three separate reports: A2, B3, and C3 | Sec. 6 | `*/report/` | `VERIFIED / COMPLETED` | `A2_Report.tex`, `B3_Report.tex`, `C3_Report.tex` drafted with all empirical data & student header |
| **REP-02** | IEEE conference style, A4 paper, double column, 10-pt font | Sec. 6 | `*/report/source/*.tex` | `VERIFIED / COMPLETED` | Templates configured with IEEEtran standard |
| **REP-03** | Max 2 pages of main content (AI Disclosure & References may extend to 3rd page) | Sec. 6 | `*/report/*.pdf` | `VERIFIED / COMPLETED` | All three compiled PDFs are strictly 2 pages each |
| **REP-04** | All mandatory sections in exact order (Intro, Problem, Data, Stats, Method, Results, Analysis, Conclusion, AI, Refs) | Sec. 6, Tab. 2 | `*/report/source/*.tex` | `VERIFIED / COMPLETED` | Verified in all 3 report sources matching Table 2 |
| **REP-05** | Every number matches notebook output with zero fabrication | Sec. 6 | `*/report/*.pdf` | `VERIFIED / COMPLETED` | Cross-verified: A2 (GBDT 4.45 MPa), B3 (LogReg 0.4890), C3 (ResNet+LSTM 21.25% BLEU-4) |
| **REP-06** | File naming convention: `<RollNo>_<TaskID>_Report.pdf` | Sec. 6 | `*/report/` | `VERIFIED / COMPLETED` | Enforced naming: `70177784_A2_Report.pdf`, `70177784_B3_Report.pdf`, `70177784_C3_Report.pdf` |

---

## 6. GitHub Repository & Submission Verification

| Requirement ID | Description | Source Section | Target Location | Status | Verification Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GIT-01** | Private repository during development | Sec. 9, Step 2 | GitHub | `READY` | Remote configured: `https://github.com/Faran70177784/AI101-ML-Assignment` |
| **GIT-02** | At least five meaningful commits per task | Sec. 9, Step 5 | Git commit history | `VERIFIED / COMPLETED` | 18 commits recorded (5 for A2, 5 for B3, 5 for C3, 3 chore/release) |
| **GIT-03** | README.md complete with student table, task links, run instructions | Sec. 9, Step 6 | `README.md` | `VERIFIED / COMPLETED` | Student details, SAP ID, report links, and run guide complete |
| **GIT-04** | Final submission tag: `git tag -a submission -m "Final submission - AI101 ML Assignment"` | Sec. 9, Step 7 | Git tags | `VERIFIED / COMPLETED` | Annotated tag `submission` created on HEAD |
| **GIT-05** | Repository made public at deadline & link verified in incognito window | Sec. 9, Step 8 | GitHub Settings | `READY` | Ready for push and public toggle at deadline |
| **GIT-06** | Link shared with Class Representative (CR) for compiled Excel sheet | Sec. 9, Step 9 | CR Sheet | `READY` | Link ready: `https://github.com/Faran70177784/AI101-ML-Assignment` |
