# AQI Forecasting for Indian Oil Corporation
### Team: Vrishank | Pranav | Harshit Garg

---

## 1. Problem Statement (What Are We Solving?)

Indian Oil Corporation (IOC) runs oil refineries that emit pollutants like SO2, NO2, PM2.5, etc. These pollutants affect the **Air Quality Index (AQI)** around the refinery zone. If the AQI rises dangerously, it harms workers and nearby residents — and the company faces regulatory action.

**The goal:** Build a model that can **predict future AQI values** using historical pollutant data, so IOC can take action *before* the air quality gets bad — not after.

---

## 2. The Dataset

**Source:** Air Quality Data in India (2015–2020)  
**Platform:** Kaggle (`rohanrao/air-quality-data-in-india`)  
**File used:** `city_day.csv` — daily readings for 26+ Indian cities

### Why this dataset?
- Covers real cities near IOC refineries (Delhi is close to Mathura refinery)
- Has **all major pollutants**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Has real missing values** — matching the problem statement ("incomplete data")
- Pre-computed AQI column already available (CPCB formula)
- 5 years of daily data — enough for seasonal pattern detection

### AQI Column Meaning (CPCB Standard)

| AQI Range | Category | Action Needed |
|-----------|----------|---------------|
| 0–50 | Good | None |
| 51–100 | Satisfactory | Minor precautions |
| 101–200 | Moderate | Sensitive groups affected |
| 201–300 | Poor | Health effects on exposure |
| 301–400 | Very Poor | Serious respiratory issues |
| 401–500 | Severe | Emergency conditions |

---

## 3. Our Pipeline (What We Did, Step by Step)

```
Raw Data → EDA → Preprocessing → Feature Selection → STL Decomposition
       → Train/Test Split → ARIMA Model → LSTM Model → Hybrid → Evaluation
```

### Step 1: Exploratory Data Analysis (EDA)
- Looked at the shape and structure of the data
- Found which columns have the most missing values
- Plotted how AQI varies across cities and over time
- **Key observation:** Delhi has high AQI (200–400+) in winter months (Oct–Jan), showing clear seasonal patterns

### Step 2: Data Preprocessing
- **Filtered to Delhi** (best data coverage, near IOC's Mathura refinery)
- **Dropped columns** with >60% missing values (some trace gases were too sparse)
- **Interpolation:** Used linear interpolation to fill gaps ≤7 days (short outages from sensor failure)
- **Forward-fill:** Filled any remaining gaps by carrying last known value forward
- This simulates what IOC would do with real-world incomplete sensor data

### Step 3: Feature Selection
Two methods used:

**Pearson Correlation** — Measures linear relationship between each pollutant and AQI
- Score of +1 means perfectly correlated (pollutant rises → AQI rises)
- Score of -1 means inversely correlated
- We kept features with |score| > 0.3

**Mutual Information** — Captures *non-linear* relationships too
- Higher score = more useful for predicting AQI
- PM2.5, PM10, and NO2 typically score highest

> **Why this matters:** Instead of feeding all 12 pollutants into the model (which adds noise), we identify which ones actually matter for AQI prediction.

### Step 4: Seasonal Decomposition (STL)
**STL = Seasonal-Trend decomposition using LOESS** (a statistical method)

It splits the AQI time series into 3 parts:
- **Trend** — Is air quality getting better or worse over years?
- **Seasonal** — Repeating weekly/annual patterns (e.g., worse on weekdays due to traffic)
- **Residual** — Random noise that can't be explained by trend or season

**Why we do this:** It proves to the teacher/audience that our data has meaningful structure, justifying the use of ARIMA (for trend/seasonality) and LSTM (for residuals).

### Step 5: Train/Test Split
- **80% training** (2015–~2019): Model learns from this
- **20% testing** (~2019–2020): We check predictions against real values
- Time order is preserved — we never shuffle (future data cannot be used to predict the past)

### Step 6: ARIMA Model
**ARIMA = AutoRegressive Integrated Moving Average**

This is a classical statistics-based time series model. It works by:
- **AR (AutoRegressive):** Uses past AQI values to predict the next one
- **I (Integrated):** Removes trend by differencing the series (makes it "stationary")
- **MA (Moving Average):** Smooths out random shocks using past forecast errors

**Parameters (p, d, q):**
- `p` = how many past values to use
- `d` = how many times to difference (usually 1 or 2)
- `q` = how many past errors to use

We use `auto_arima` to automatically find the best (p, d, q) using **AIC score** (lower = better).

**Limitation:** ARIMA is linear — it can't capture sudden pollution spikes during industrial accidents or dust storms.

### Step 7: LSTM Model
**LSTM = Long Short-Term Memory** (a type of deep learning / neural network)

LSTMs are designed for sequential data (like time series). They have memory cells that can "remember" information from far back in time.

**Architecture we used:**
```
LSTM(64 units) → Dropout(0.2) → LSTM(32 units) → Dropout(0.2) → Dense(1 output)
```

**What LSTM learns in our hybrid:** It doesn't predict AQI directly. Instead, it learns to predict ARIMA's **errors (residuals)** — the pollution patterns ARIMA missed (non-linear spikes, complex interactions between pollutants).

**Hyperparameter Tuning** (done by Pranav):
We tested different combinations:
- `units`: [64, 128] — how many neurons per LSTM layer
- `dropout`: [0.1, 0.2] — randomly turns off some neurons during training to prevent overfitting

The combination with the lowest **validation loss (MSE)** is chosen as the final model.

### Step 8: Hybrid ARIMA + LSTM
**Formula:**

```
Final Prediction = ARIMA Forecast + LSTM Residual Correction
```

- ARIMA handles the **predictable linear part** (trend + weekly seasonality)
- LSTM corrects the **unpredictable non-linear part** (sudden pollution events)

This is why the hybrid consistently outperforms both individual models.

### Step 9: Evaluation
Three metrics computed on the test set:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **RMSE** | √(mean(y - ŷ)²) | Penalises large errors heavily |
| **MAE** | mean(|y - ŷ|) | Average absolute error (in AQI units) |
| **MAPE** | mean(|y - ŷ| / y) × 100 | Error as % of actual AQI |

Lower is always better for all three.

---

## 4. Team Roles

| Name | What They Did |
|------|--------------|
| **Vrishank** | EDA (Section 2), STL Decomposition (Section 5), ARIMA Model (Section 7) |
| **Pranav** | LSTM Model architecture, Hyperparameter grid search (Section 8) |
| **Harshit Garg** | Data Preprocessing (Section 3), Feature Selection (Section 4), Evaluation metrics & comparison charts (Section 10) |

---

## 5. Key Concepts: Quick Explanation for Q&A

**Q: Why ARIMA AND LSTM? Why not just one?**  
A: ARIMA is great at linear patterns (trends, weekly cycles) but fails on sudden spikes. LSTM handles non-linear patterns but needs a lot of data and may miss long-term trends. Together, each covers the other's weakness.

**Q: What is a "residual"?**  
A: The residual is the difference between what ARIMA predicted and what actually happened. If ARIMA predicted AQI = 150 but actual was 180, the residual is 30. LSTM learns to predict these corrections.

**Q: Why STL decomposition?**  
A: It helps us understand the data and confirm seasonality exists before modelling. Also, it visually demonstrates to stakeholders that the data has predictable structure.

**Q: What does "hyperparameter tuning" mean?**  
A: Neural networks have settings (like how many neurons, how much dropout) that can't be learned from data — they must be set manually. We try different combinations and pick whichever gives the best performance on a validation set.

**Q: Why not use the most recent data (2024/2025)?**  
A: The dataset we used covers 2015–2020. In a real deployment, IOC would use live sensor feeds and retrain monthly. For our academic project, this dataset is sufficient to demonstrate the methodology.

**Q: What is overfitting?**  
A: When a model memorises training data instead of learning general patterns — it performs well in training but poorly on new data. We prevent this with `Dropout` layers and `EarlyStopping` in the LSTM.

---

## 6. Results Summary

The Hybrid ARIMA + LSTM model achieves:
- **Lower RMSE** than standalone ARIMA and standalone LSTM
- **Lower MAPE** — predictions are within a smaller percentage of actual AQI
- Visual plots show hybrid predictions track the actual AQI curve more closely, especially during pollution spikes

> Typical improvement: Hybrid reduces RMSE by **15–30%** compared to ARIMA alone (literature benchmark; actual numbers from the notebook output).

---

## 7. Real-World Application for IOC

If Indian Oil Corporation deployed this system:

1. **Daily forecast dashboard** — Operations team sees predicted AQI for next 7 days around the refinery
2. **Automatic alerts** — If forecast exceeds 200 (Poor category), alert is sent to emission control team
3. **Refinery scheduling** — Flaring, maintenance, or high-emission processes are scheduled on low-AQI forecast days
4. **Regulatory reporting** — Predicted exceedance events are documented before they happen, showing CPCB proactive compliance
5. **Model updates** — Each week, new sensor data is added and the model is fine-tuned (incremental learning)

---

## 8. Files in This Project

| File | Description |
|------|-------------|
| `AQI_Forecasting.ipynb` | Main Jupyter notebook with all code and plots |
| `PROJECT_EXPLANATION.md` | This file — project guide for the team |
| `air-quality-data-in-india/city_day.csv` | Dataset (downloaded by the notebook) |

---

## 9. How to Run the Notebook

1. Open `AQI_Forecasting.ipynb` in Jupyter Notebook or Google Colab
2. Run Cell 2 (install packages) — wait for completion
3. Run Cell 3 (imports)
4. Run Cell 5 (data download) — enter your Kaggle username and API key when prompted
5. Run all remaining cells in order (Kernel → Restart & Run All)

**Expected runtime:** ~15–25 minutes (auto_arima is the slowest step at ~5 min)

---

*Project for AI Course — Academic Year 2025–2026*
