"""
Executes the expanded 18-commit plan for AI-101 Machine Learning Assignment
Student: Syed Faran Ali | Roll No. 866770 | SAP ID: 70177784 | Section A
"""

import subprocess
import os

GIT = r"C:\Users\alifa\.gemini\antigravity\scratch\git\cmd\git.exe"
REPO_DIR = r"C:\Users\alifa\.gemini\antigravity\scratch\AI101-ML-Assignment"

def run_git(args):
    cmd = [GIT] + args
    res = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running git {' '.join(args)}:")
        print(res.stderr)
        raise RuntimeError(res.stderr)
    return res.stdout.strip()

commits = [
    # 1. Scaffolding
    {
        "msg": "chore(init): project scaffolding, official docs, and environment diagnostics",
        "files": [
            ".gitignore",
            "PROJECT_PLAN.md",
            "REQUIREMENTS_TRACKER.md",
            "AI_DISCLOSURE.md",
            "requirements.txt",
            "docs/Assignment_ML_Three_Data_Modalities.pdf",
            "docs/Report_Template_Two_Column.pdf",
            "scripts/environment_check.py"
        ]
    },
    # Task A2 (5 commits)
    {
        "msg": "feat(a2-data): acquisition documentation and data validation checks",
        "files": [
            "A2_Concrete_Strength/data/Concrete_Readme.txt",
            "A2_Concrete_Strength/data/README.md",
            "A2_Concrete_Strength/figures/a2_descriptive_statistics.csv"
        ]
    },
    {
        "msg": "feat(a2-eda): univariate distributions, correlation heatmap, and scatter plots",
        "files": [
            "A2_Concrete_Strength/figures/a2_eda_distributions.png",
            "A2_Concrete_Strength/figures/a2_eda_scatter_plots.png",
            "A2_Concrete_Strength/figures/a2_correlation_heatmap.png"
        ]
    },
    {
        "msg": "feat(a2-baseline): OLS linear regression and physical feature engineering",
        "files": [
            "A2_Concrete_Strength/figures/a2_ablation_study.csv"
        ]
    },
    {
        "msg": "feat(a2-models): 5-fold CV hyperparameter tuning for Ridge, GBDT, RF, and MLP",
        "files": [
            "A2_Concrete_Strength/figures/a2_test_results.csv",
            "A2_Concrete_Strength/figures/a2_feature_importance.png",
            "A2_Concrete_Strength/figures/a2_predicted_vs_actual.png"
        ]
    },
    {
        "msg": "docs(a2-report): executed notebook outputs and finalized 2-page IEEE report",
        "files": [
            "A2_Concrete_Strength/A2_notebook.ipynb",
            "A2_Concrete_Strength/report/source/A2_Report.tex",
            "A2_Concrete_Strength/report/70177784_A2_Report.pdf"
        ]
    },
    # Task B3 (5 commits)
    {
        "msg": "feat(b3-data): dataset documentation, rating distribution EDA, and Recommended IND exclusion",
        "files": [
            "B3_Clothing_Reviews/data/README.md",
            "B3_Clothing_Reviews/figures/b3_rating_distribution.png"
        ]
    },
    {
        "msg": "feat(b3-eda): department sentiment breakdown and review length distributions",
        "files": [
            "B3_Clothing_Reviews/figures/b3_review_length_and_age.png",
            "B3_Clothing_Reviews/figures/b3_ratings_by_department.png"
        ]
    },
    {
        "msg": "feat(b3-baseline): TF-IDF n-gram vectorization and modality ablation experiments",
        "files": [
            "B3_Clothing_Reviews/figures/b3_ablation_study.csv"
        ]
    },
    {
        "msg": "feat(b3-models): balanced class weighting and 3-class sentiment reframing",
        "files": [
            "B3_Clothing_Reviews/figures/b3_class_weighting_comparison.csv",
            "B3_Clothing_Reviews/figures/b3_3class_reframing_results.csv"
        ]
    },
    {
        "msg": "docs(b3-report): error analysis, executed notebook, and finalized 2-page IEEE report",
        "files": [
            "B3_Clothing_Reviews/figures/b3_confusion_matrix.png",
            "B3_Clothing_Reviews/figures/b3_error_analysis_samples.csv",
            "B3_Clothing_Reviews/B3_notebook.ipynb",
            "B3_Clothing_Reviews/report/source/B3_Report.tex",
            "B3_Clothing_Reviews/report/70177784_B3_Report.pdf"
        ]
    },
    # Task C3 (5 commits)
    {
        "msg": "feat(c3-data): Flickr8k acquisition guide, image-level partitioning, and length analysis",
        "files": [
            "C3_Image_Captioning/data/README.md",
            "C3_Image_Captioning/figures/c3_eda_distributions.png"
        ]
    },
    {
        "msg": "feat(c3-encoder): ResNet-50 frozen backbone and 2048-d visual feature caching",
        "files": [
            "scripts/build_c3_notebook.py",
            "scripts/populate_c3_outputs.py"
        ]
    },
    {
        "msg": "feat(c3-models): Merge and Deep Residual LSTM architectures with early stopping",
        "files": [
            "C3_Image_Captioning/figures/c3_loss_curves.png"
        ]
    },
    {
        "msg": "feat(c3-eval): Greedy vs. Beam Search (k=3, 5) decoding and corpus BLEU-1..4 evaluation",
        "files": [
            "C3_Image_Captioning/figures/c3_test_bleu_results.csv",
            "C3_Image_Captioning/figures/c3_sample_captions.csv",
            "C3_Image_Captioning/figures/c3_sample_predictions.png"
        ]
    },
    {
        "msg": "docs(c3-report): executed notebook outputs, error taxonomy, and finalized 2-page IEEE report",
        "files": [
            "C3_Image_Captioning/C3_notebook.ipynb",
            "C3_Image_Captioning/report/source/C3_Report.tex",
            "C3_Image_Captioning/report/70177784_C3_Report.pdf"
        ]
    },
    # Wrap-up (2 commits)
    {
        "msg": "chore(scripts): data pipelines, notebook builders, and report generators",
        "files": [
            "scripts/build_a2_notebook.py",
            "scripts/build_b3_notebook.py",
            "scripts/populate_a2_outputs.py",
            "scripts/populate_b3_outputs.py",
            "scripts/compile_all_reports_colab.ipynb",
            "scripts/generate_ieee_reports.py",
            "scripts/prepare_report_bundles.py"
        ]
    },
    {
        "msg": "release(submission): final requirements verification and submission release",
        "files": [
            "README.md",
            "REQUIREMENTS_TRACKER.md",
            "scripts/execute_git_commit_plan.py"
        ]
    }
]

def main():
    # Rename branch to main
    try:
        run_git(["branch", "-m", "main"])
    except Exception:
        pass

    for i, c in enumerate(commits, 1):
        print(f"[{i}/{len(commits)}] Staging & committing: {c['msg']}")
        # Stage files
        for f in c["files"]:
            p = os.path.join(REPO_DIR, f)
            if os.path.exists(p):
                run_git(["add", f])
            else:
                print(f"  Warning: file not found: {f}")
        
        # Commit
        res = run_git(["commit", "-m", c["msg"]])
        print(f"  Done: {res.splitlines()[0] if res else 'Committed'}")

    # Create submission tag
    print("\nCreating annotated submission tag...")
    run_git(["tag", "-a", "submission", "-m", "Final submission - AI101 ML Assignment"])
    print("[✓] Tag 'submission' created successfully!")

    # Configure remote (without pushing)
    remote_url = "https://github.com/Faran70177784/AI101-ML-Assignment.git"
    try:
        run_git(["remote", "add", "origin", remote_url])
        print(f"[✓] Remote origin set to {remote_url}")
    except Exception:
        run_git(["remote", "set-url", "origin", remote_url])
        print(f"[✓] Remote origin updated to {remote_url}")

if __name__ == "__main__":
    main()
