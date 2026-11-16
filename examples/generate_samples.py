"""Generate realistic sample datasets for testing & Reels demos."""

from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# 1. Telco Churn (Classification, EDA, Clean, Metrics, SHAP, PCA)
n = 1200
tenure = rng.integers(1, 72, n)
monthly_charge = rng.uniform(25, 115, n).round(2)
support_tickets = rng.poisson(1.4, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
usage_gb = rng.normal(45, 18, n).clip(2, 120).round(1)
logit = -1.8 + 0.03 * monthly_charge - 0.04 * tenure + 0.45 * support_tickets + np.where(contract == "Month-to-month", 1.1, -0.6)
prob = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < prob).astype(int)

df_churn = pd.DataFrame({
    "customer_id": [f"CUST_{i:04d}" for i in range(n)],
    "tenure_months": tenure,
    "monthly_charges": monthly_charge,
    "support_tickets": support_tickets,
    "contract_type": contract,
    "avg_gb_used": usage_gb,
    "constant_flag": 1,
    "churn": churn,
})
# Inject realistic missingness + 15 duplicates
df_churn.loc[rng.choice(n, 45, replace=False), "avg_gb_used"] = np.nan
df_churn.loc[rng.choice(n, 20, replace=False), "contract_type"] = np.nan
df_churn = pd.concat([df_churn, df_churn.iloc[:15]], ignore_index=True)
df_churn.to_csv(OUT / "churn_sample.csv", index=False)

# 2. Drifted Current Dataset (for /drift)
df_curr = df_churn.iloc[:800].copy()
df_curr["monthly_charges"] = df_curr["monthly_charges"] * 1.25 + 12  # Clear price drift
df_curr["support_tickets"] = df_curr["support_tickets"] + rng.poisson(1.2, len(df_curr))
df_curr.to_csv(OUT / "churn_current_drifted.csv", index=False)

# 3. Housing Price (for /regression)
m = 1000
sqft = rng.normal(1800, 500, m).clip(600, 4500).round(0)
bedrooms = rng.integers(1, 6, m)
age = rng.integers(0, 60, m)
dist_center = rng.uniform(0.5, 25, m).round(1)
price = 80000 + 185 * sqft + 15000 * bedrooms - 1200 * age - 4500 * dist_center + rng.normal(0, 22000, m)
pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "house_age": age,
    "dist_to_center_km": dist_center,
    "price_usd": price.round(0),
}).to_csv(OUT / "housing_regression.csv", index=False)

# 4. A/B Test (for /abtest)
na, nb = 2500, 2520
ab_df = pd.DataFrame({
    "group": ["control"] * na + ["variant"] * nb,
    "converted": np.concatenate([rng.binomial(1, 0.112, na), rng.binomial(1, 0.136, nb)]),
})
ab_df.to_csv(OUT / "abtest_checkout.csv", index=False)

# 5. Time Series (for /forecast)
dates = pd.date_range("2025-01-01", periods=120, freq="D")
trend = np.linspace(100, 185, 120)
seasonal = 22 * np.sin(2 * np.pi * np.arange(120) / 7)
noise = rng.normal(0, 6, 120)
pd.DataFrame({"date": dates, "daily_orders": (trend + seasonal + noise).round(1)}).to_csv(
    OUT / "daily_orders_ts.csv", index=False
)

# 6. User activity events (for /ds-cohort)
users = 900
signup = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, users), unit="D")
rows = []
for uid, sd in enumerate(signup):
    churn_rate = rng.uniform(0.08, 0.35)
    month = 0
    while True:
        n_ev = rng.poisson(3) + 1
        for _ in range(n_ev):
            d = sd + pd.Timedelta(days=int(month * 30 + rng.integers(0, 30)))
            rows.append((f"U{uid:04d}", d.date(), round(float(rng.gamma(2.0, 18.0)), 2)))
        month += 1
        if rng.random() < churn_rate or month > 8:
            break
pd.DataFrame(rows, columns=["user_id", "event_date", "revenue"]).to_csv(OUT / "user_events.csv", index=False)

# 7. Marketing campaign (for /ds-uplift): treatment works mostly for young, low-tenure users
k = 8000
age_u = rng.integers(18, 70, k)
tenure_u = rng.integers(1, 60, k)
visits = rng.poisson(4, k)
segment = rng.choice(["new", "regular", "vip"], k, p=[0.4, 0.45, 0.15])
treat = rng.binomial(1, 0.5, k)
base = -2.2 + 0.08 * visits + np.where(segment == "vip", 0.6, 0.0)
lift = treat * (1.3 * (age_u < 35) + 0.6 * (tenure_u < 12) - 0.35)
p_conv = 1 / (1 + np.exp(-(base + lift)))
pd.DataFrame({
    "age": age_u, "tenure_months": tenure_u, "site_visits": visits, "segment": segment,
    "treatment": treat, "converted": rng.binomial(1, p_conv),
}).to_csv(OUT / "campaign_uplift.csv", index=False)

# 8. Product reviews (for /ds-text), Turkish
pos = ["harika ürün çok memnunum", "kargo hızlı geldi teşekkürler", "kalitesi mükemmel tavsiye ederim",
       "fiyatına göre çok iyi", "beklediğimden güzel çıktı", "paketleme özenliydi süper"]
neg = ["ürün bozuk geldi iade ettim", "kargo çok geç geldi", "kalitesi berbat param boşa gitti",
       "açıklamadaki gibi değil hayal kırıklığı", "müşteri hizmetleri ilgisiz", "bir hafta sonra kırıldı"]
products = ["kulaklık", "telefon kılıfı", "ayakkabı", "mont", "kahve makinesi", "laptop çantası", "saat", "matara"]
filler = ["", " ama", " yine de", " kesinlikle", " gerçekten", " açıkçası"]
texts, labels = [], []
for i in range(1500):
    is_pos = rng.random() < 0.62
    pool = pos if is_pos else neg
    parts = list(rng.choice(pool, rng.integers(1, 3), replace=False))
    if rng.random() < 0.25:  # mixed opinion
        parts.append(str(rng.choice(neg if is_pos else pos)))
    t = " ".join(parts) + str(rng.choice(filler)) + str(rng.choice(["", " " + str(rng.choice(products))]))
    texts.append(t.strip())
    if rng.random() < 0.06:  # label noise
        is_pos = not is_pos
    labels.append("olumlu" if is_pos else "olumsuz")
pd.DataFrame({"review": texts, "sentiment": labels, "stars": [int(rng.integers(4, 6)) if l == "olumlu" else int(rng.integers(1, 3)) for l in labels]}).to_csv(
    OUT / "reviews_tr.csv", index=False
)
print(f"Generated 8 sample datasets in {OUT}")
