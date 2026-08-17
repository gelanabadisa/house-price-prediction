# 🏠 House Price Prediction Dashboard

A machine learning project that predicts house prices from property features
(area, bedrooms, bathrooms, amenities, etc.) and serves predictions through an
interactive **Streamlit** dashboard.

## How it works

- `train_model.py` trains and compares three regression algorithms
  (Linear Regression, Random Forest, Gradient Boosting) using 5-fold
  cross-validation, then automatically saves the best-performing pipeline
  (preprocessing + model bundled together) to `models/house_price_model.pkl`.
- `app.py` loads that saved pipeline and lets a user enter property details
  through a form, then displays the predicted price.

**Primary algorithm:** Random Forest Regressor — a strong default for this
kind of tabular data because it handles the mix of numeric and categorical
features without manual feature engineering, captures non-linear
relationships between area/rooms and price, and needs no feature scaling.
The training script benchmarks it against Linear Regression and Gradient
Boosting and keeps whichever wins on test R².

## Project structure

```
house-price-prediction/
├── app.py                  # Streamlit dashboard
├── train_model.py          # Training script
├── requirements.txt
├── .gitignore
├── data/
│   └── Housing_sample.csv  # small sample — replace with your full dataset
└── models/                 # created after training (model + metadata)
```

---

## Step 1 — Set up the project locally

```bash
# 1. Create a project folder and move into it (or unzip the files you downloaded)
cd house-price-prediction

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Step 2 — Add your full dataset

Replace the sample file with your full dataset, keeping the same column
names (`price, area, bedrooms, bathrooms, stories, mainroad, guestroom,
basement, hotwaterheating, airconditioning, parking, prefarea,
furnishingstatus`):

```bash
cp /path/to/your/Housing.csv data/Housing.csv
```

## Step 3 — Train the model

```bash
python train_model.py --data data/Housing.csv
```

This prints R² / MAE / RMSE for each algorithm, then saves the winner to
`models/house_price_model.pkl` and `models/metadata.json`.

## Step 4 — Run the dashboard

```bash
streamlit run app.py
```

Streamlit will open the dashboard in your browser (usually
`http://localhost:8501`). Enter property details and click **Predict Price**.

---

## Step 5 — Publish the code to GitHub (public repo)

```bash
# 1. Initialize git in the project folder
git init

# 2. Stage and commit your files
git add .
git commit -m "Initial commit: house price prediction app"

# 3. Create a new PUBLIC repo on GitHub
#    Go to https://github.com/new, name it e.g. "house-price-prediction",
#    set visibility to Public, do NOT initialize with a README (you already have one).

# 4. Link your local repo to GitHub and push
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-prediction.git
git push -u origin main
```

If you'd rather use the GitHub CLI instead of the website:

```bash
gh repo create house-price-prediction --public --source=. --remote=origin --push
```

### A note on the dataset in your public repo

If your full `Housing.csv` is from a public source (e.g. Kaggle) and license
permits redistribution, it's fine to commit it. If you're not sure of the
license, don't commit the full file — keep only `data/Housing_sample.csv`
in the repo and add `data/Housing.csv` to `.gitignore`, with a note in the
README telling visitors where to download the full dataset.

---

## Step 6 (optional) — Deploy it live for your portfolio

Streamlit Community Cloud (free) will host the dashboard and give you a
public link to put in your portfolio/resume:

1. Push the repo to GitHub (Step 5).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select your repo, branch `main`, and file `app.py`.
4. Click **Deploy**. You'll get a live URL like
   `https://<your-app-name>.streamlit.app`.

Make sure `models/house_price_model.pkl` and `models/metadata.json` are
committed to the repo (or trained via a startup script) so the deployed app
has a model to load — Streamlit Cloud won't run `train_model.py`
automatically.

---

## License

Feel free to use the MIT License for a portfolio project — add a
`LICENSE` file with the standard MIT text if you'd like others to be able
to reuse your code.
