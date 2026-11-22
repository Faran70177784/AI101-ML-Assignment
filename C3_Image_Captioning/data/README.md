# Dataset & Execution Guide: Task C3 (Flickr8k Image Captioning)

## 1. Dataset Overview
- **Dataset Source:** Kaggle: `adityajn105/flickr8k`
- **Official References:** Hodosh, Young, and Hockenmaier (2013), "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", *Journal of Artificial Intelligence Research*, 47, 853-899.
- **Dataset Scope:** 8,091 images with 5 human reference captions per image (40,455 captions total).
- **Target Task:** Supervised image-to-text sequence-to-sequence generation ($I \to Y$).

---

## 2. Mandatory Image-Level Splitting

To eliminate visual data leakage into the evaluation sets, splitting is enforced strictly at the **image level**:
- **Train Split:** 6,000 images (30,000 captions)
- **Validation Split:** 1,000 images (5,000 captions)
- **Test Split:** 1,091 images (5,455 captions)

> **Rule:** All five captions for any given image remain in the same split. Test images are never seen during encoder feature extraction or decoder training.

---

## 3. How to Execute `C3_notebook.ipynb` in Google Colab (GPU Accelerated)

Because deep sequence modeling with ResNet-50 feature caching and LSTM sequence training requires GPU compute, `C3_notebook.ipynb` is configured for **Google Colab with free T4 GPU acceleration**.

### Step 1: Open Google Colab with GPU
1. In your browser, open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `C3_Image_Captioning/C3_notebook.ipynb` from this folder.
3. In the Colab menu bar, click **Runtime** $\to$ **Change runtime type** $\to$ select **T4 GPU** $\to$ click **Save**.

### Step 2: Run All Cells
1. Click **Runtime** $\to$ **Run all** (or press `Ctrl + F9`).
2. The notebook is fully automated:
   - Cell 1 checks the T4 GPU device.
   - Cell 2 automatically downloads the Flickr8k dataset via `kagglehub` or direct fallback.
   - Cells 3–5 perform caption cleaning, vocabulary construction (min-frequency $\ge 5$), and length analysis.
   - Cells 6–7 extract and cache 2048-d visual feature vectors using a frozen ImageNet ResNet-50 backbone.
   - Cells 8–11 train **Model 1 ("Merge" Architecture)** and **Model 2 (Deep Residual Architecture)** with early stopping.
   - Cells 12–15 run Greedy and Beam Search ($k = 3, 5$) decoding across all 1,091 test images, computing Corpus BLEU-1 to BLEU-4 against all 5 reference captions.
   - Entire execution takes **approx. 8–12 minutes** on a free T4 GPU.

### Step 3: Save Executed Outputs Back to Project
1. **Download the executed notebook:**
   - In Colab: Click **File** $\to$ **Download** $\to$ **Download .ipynb**.
   - Save/replace it at `C3_Image_Captioning/C3_notebook.ipynb` so all cell outputs, training curves, and BLEU tables are preserved.
2. **Retrieve Figures and Metrics:**
   - The final cell automatically creates and downloads **`c3_artifacts.zip`**.
   - Extract `c3_artifacts.zip` into `C3_Image_Captioning/figures/`.
   - It contains:
     - `c3_eda_distributions.png`
     - `c3_loss_curves.png`
     - `c3_sample_predictions.png`
     - `c3_test_bleu_results.csv`
     - `c3_sample_captions.csv`
