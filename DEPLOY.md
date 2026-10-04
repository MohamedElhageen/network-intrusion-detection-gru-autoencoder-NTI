# Deploying the Streamlit App (Streamlit Community Cloud)

This puts the app online at a free public link like `https://your-app-name.streamlit.app`, so it opens from any computer without installing anything.

**Time needed:** ~15 minutes (most of it is the first build).
**You need:** a GitHub account (free) and this project folder.

---

## What gets uploaded (and what doesn't)

The app needs only about **10 MB**:

| Uploaded | Size | Why |
|---|---|---|
| `app/` | 70 KB | the two pages, the styling, and `app/requirements.txt` |
| `src/` | 240 KB | the model and scoring code the app imports |
| `models/` | 3.9 MB | trained GRU, Isolation Forest, preprocessor, thresholds |
| `data/demo/` | 4.9 MB | the attack replay files |
| `results/` | 5 KB | metric tables (the Inside page reads the forest's alarm line from here) |
| `.streamlit/config.toml` | 1 KB | the custom colour theme |
| `notebooks/`, `tests/`, `README.md` | ~0.8 MB | not needed by the app, but nice to have in the repo |

**Not uploaded** (already in `.gitignore`): `data/raw/` (2 GB), `data/processed/`, virtual environments, `.rar` files.

`app/requirements.txt` is a **slim** dependency list for the server (CPU-only PyTorch, no Jupyter or pytest). Streamlit Cloud uses it automatically because it sits next to `streamlit_app.py`. Your main `requirements.txt` is unchanged for local development.

---

## Step 1: Put the project on GitHub

### 1a. Create an empty repository
1. Go to **github.com**, sign in, click **+ → New repository**.
2. Name it, e.g. `network-intrusion-detector`.
3. Choose **Public** (simplest) or **Private** (also works; Streamlit will ask for permission to read it).
4. **Don't** tick "Add a README" or ".gitignore"; the project already has them.
5. Click **Create repository** and copy the URL it shows (`https://github.com/<you>/network-intrusion-detector.git`).

### 1b. Upload the project from your computer
Open a terminal in the project folder (`Final_project`) and run these one at a time:

```bash
git init -b main
```
```bash
git add .
```
```bash
git commit -m "Network intrusion detector: models, app and notebooks"
```
```bash
git remote add origin https://github.com/<you>/network-intrusion-detector.git
```
```bash
git push -u origin main
```

The first push opens a GitHub sign-in window. Log in there.

**Check:** on the GitHub page you should see `app/`, `src/`, `models/`, `data/demo/`, but **no** `data/raw/` CSVs.

---

## Step 2: Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and click **Continue with GitHub**. Approve the access it asks for.
2. Click **Create app**, then **"Deploy a public app from GitHub"** (or "Yup, I have an app").
3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `<you>/network-intrusion-detector` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |
   | App URL (optional) | e.g. `nti-intrusion-detector` |
4. Open **Advanced settings** and set **Python version: 3.12**.
5. Click **Deploy**.

The first build takes **about 5–10 minutes** (installing PyTorch is the slow part). You'll see the install log on the right. When it finishes, the app opens automatically.

---

## Step 3: Check it works
- **Live Detector:** pick **DDoS**, click **Start live replay**. You should get the first alert at window 30, and 20/25 attack windows caught with the Ensemble.
- **Inside the Model:** the heatmaps and the ensemble breakdown should appear for window 30.

---

## Updating the app later
Any change you push to GitHub redeploys automatically:
```bash
git add .
```
```bash
git commit -m "describe the change"
```
```bash
git push
```

---

## Before the presentation
- **Free apps go to sleep** after a period without visitors. Open the link **15–30 minutes before** presenting. If you see "This app has gone to sleep", click **"Yes, get this app back up!"** and wait about a minute.
- **Keep the local version as a backup** (`py -m streamlit run app/streamlit_app.py`) in case the venue's internet is unreliable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails while installing `torch` | Make sure **Python 3.12** is selected (Advanced settings). Then **⋮ → Reboot app**. |
| `ModuleNotFoundError: No module named 'src'` | The main file path must be exactly `app/streamlit_app.py`, and `src/` must be in the repo. |
| "Model files not found in models/" | `models/` wasn't pushed. Check it's on GitHub; run `git add models` and push. |
| "No demo streams in data/demo/" | `data/demo/` wasn't pushed. Same fix with `git add data/demo`. |
| Warning about scikit-learn versions | Harmless, but `app/requirements.txt` pins `scikit-learn==1.9.1` (the version that saved the forest) to avoid it. |
| App is very slow or restarts ("Oh no.") | Free apps have about 1 GB of memory. Reboot from **⋮ → Reboot app**; avoid uploading very large CSVs. |
| Colours look like default Streamlit | `.streamlit/config.toml` wasn't pushed (hidden folder). Run `git add .streamlit` and push. |

---

## Alternative: Hugging Face Spaces
If Streamlit Cloud gives you trouble, **huggingface.co/spaces** also hosts Streamlit apps for free (choose the **Docker → Streamlit** template). It needs a small `Dockerfile`, so Streamlit Cloud is the simpler first choice.
