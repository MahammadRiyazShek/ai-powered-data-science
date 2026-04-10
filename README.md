# AI-Powered Data Science Studio

This repository is a flat-file project designed to be uploaded directly to GitHub and deployed with GitHub Pages.

## What it includes

- A working salary prediction audit built in the browser using linear regression.
- A working customer segmentation dashboard built in the browser using K-Means clustering.
- Dirty sample CSV datasets for both workflows.
- Python audit scripts that mirror the project logic for local execution.
- A polished landing page that works without any backend.

## Root files only

All files are intentionally kept at the repository root so you can upload them directly without managing nested folders.

## Deploy to GitHub Pages

1. Create a new GitHub repository.
2. Upload all files from this package to the root of the repository.
3. Commit and push.
4. In GitHub, open **Settings → Pages**.
5. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
6. Select your main branch and the **/(root)** folder.
7. Save. Your site will deploy on GitHub Pages.

## Run the Python audits locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python salary_prediction_audit.py
python customer_segmentation_audit.py
```

## Browser workflow

Open `index.html` locally for a quick preview or deploy it on GitHub Pages for a shareable live project. The page supports:

- sample data loading
- CSV uploads
- live metrics
- charts
- prediction for 15 years of experience
- elbow analysis
- silhouette score
- new customer classification
