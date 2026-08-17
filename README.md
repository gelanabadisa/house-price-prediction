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
