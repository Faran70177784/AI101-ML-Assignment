"""
Prepares standalone Overleaf packages and a Colab compilation notebook for Task A2, B3, and C3 IEEE reports.
Student: Syed Faran Ali
Roll No: 866770 | SAP ID: 70177784 | Section A
"""

import os
import shutil
import zipfile
import json

ROOT_DIR = r"C:\Users\alifa\.gemini\antigravity\scratch\AI101-ML-Assignment"
SAP_ID = "70177784"

def prepare_bundles():
    tasks = [
        {
            "id": "A2",
            "tex_path": os.path.join(ROOT_DIR, "A2_Concrete_Strength", "report", "source", "A2_Report.tex"),
            "fig_dir": os.path.join(ROOT_DIR, "A2_Concrete_Strength", "figures"),
            "figs": ["a2_eda_scatter_plots.png", "a2_predicted_vs_actual.png", "a2_eda_distributions.png"],
            "out_zip": os.path.join(ROOT_DIR, "A2_Concrete_Strength", "report", f"{SAP_ID}_A2_Report_Package.zip")
        },
        {
            "id": "B3",
            "tex_path": os.path.join(ROOT_DIR, "B3_Clothing_Reviews", "report", "source", "B3_Report.tex"),
            "fig_dir": os.path.join(ROOT_DIR, "B3_Clothing_Reviews", "figures"),
            "figs": ["b3_rating_distribution.png", "b3_confusion_matrix.png"],
            "out_zip": os.path.join(ROOT_DIR, "B3_Clothing_Reviews", "report", f"{SAP_ID}_B3_Report_Package.zip")
        },
        {
            "id": "C3",
            "tex_path": os.path.join(ROOT_DIR, "C3_Image_Captioning", "report", "source", "C3_Report.tex"),
            "fig_dir": os.path.join(ROOT_DIR, "C3_Image_Captioning", "figures"),
            "figs": ["c3_loss_curves.png", "c3_sample_predictions.png"],
            "out_zip": os.path.join(ROOT_DIR, "C3_Image_Captioning", "report", f"{SAP_ID}_C3_Report_Package.zip")
        }
    ]

    for t in tasks:
        with open(t["tex_path"], "r", encoding="utf-8") as f:
            tex_content = f.read()

        # Create self-contained version where figure paths are local
        local_tex = tex_content.replace("../../figures/", "")

        with zipfile.ZipFile(t["out_zip"], "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{SAP_ID}_{t['id']}_Report.tex", local_tex)
            for fig in t["figs"]:
                fig_path = os.path.join(t["fig_dir"], fig)
                if os.path.exists(fig_path):
                    zf.write(fig_path, arcname=fig)
                    print(f"Added {fig} to {t['out_zip']}")
                else:
                    print(f"Warning: Figure {fig_path} not found!")

        print(f"Created self-contained package: {t['out_zip']}")

    # Also build compile_all_reports_colab.ipynb
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# IEEE Report Compiler for Tasks A2, B3, C3\n",
                    f"**Student:** Syed Faran Ali  \n",
                    f"**Roll Number:** 866770 | **SAP ID:** {SAP_ID} | **Section:** Section A  \n",
                    f"**Course:** AI-101 Introduction to Artificial Intelligence, The University of Lahore  \n\n",
                    f"This notebook installs `texlive` and compiles all three IEEE 2-column reports into:\n",
                    f"- `{SAP_ID}_A2_Report.pdf`\n",
                    f"- `{SAP_ID}_B3_Report.pdf`\n",
                    f"- `{SAP_ID}_C3_Report.pdf`\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Install TeX Live LaTeX Compiler\n",
                    "!apt-get update -qq\n",
                    "!apt-get install -y -qq texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-publishers > /dev/null\n",
                    "print(\"TeX Live installed successfully!\")\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 2: Upload Report Zip Packages\n",
                    "from google.colab import files\n",
                    "import os, zipfile, subprocess\n\n",
                    "print(\"Please upload the 3 report packages:\")\n",
                    f"print(\"1. {SAP_ID}_A2_Report_Package.zip\")\n",
                    f"print(\"2. {SAP_ID}_B3_Report_Package.zip\")\n",
                    f"print(\"3. {SAP_ID}_C3_Report_Package.zip\")\n",
                    "uploaded = files.upload()\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 3: Extract and Compile PDFs\n",
                    f"sap_id = '{SAP_ID}'\n",
                    "tasks = ['A2', 'B3', 'C3']\n\n",
                    "for t in tasks:\n",
                    "    zip_name = f\"{sap_id}_{t}_Report_Package.zip\"\n",
                    "    folder = f\"build_{t}\"\n",
                    "    os.makedirs(folder, exist_ok=True)\n",
                    "    if zip_name in uploaded:\n",
                    "        with zipfile.ZipFile(zip_name, 'r') as zf:\n",
                    "            zf.extractall(folder)\n",
                    "        print(f\"Compiling {sap_id}_{t}_Report.pdf...\")\n",
                    "        cmd = f\"cd {folder} && pdflatex -interaction=nonstopmode {sap_id}_{t}_Report.tex && pdflatex -interaction=nonstopmode {sap_id}_{t}_Report.tex\"\n",
                    "        subprocess.run(cmd, shell=True, check=True)\n",
                    "        pdf_out = f\"{folder}/{sap_id}_{t}_Report.pdf\"\n",
                    "        if os.path.exists(pdf_out):\n",
                    "            print(f\"[✓] Compiled successfully: {pdf_out}\")\n",
                    "            files.download(pdf_out)\n",
                    "        else:\n",
                    "            print(f\"[!] Failed to generate {pdf_out}\")\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    colab_nb_path = os.path.join(ROOT_DIR, "scripts", "compile_all_reports_colab.ipynb")
    with open(colab_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created Colab compilation notebook: {colab_nb_path}")

if __name__ == "__main__":
    prepare_bundles()
