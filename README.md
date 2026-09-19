# 🌧️ RainProof — Parametric Protection Simulator Lab

> **Tagline:** Design weather-income protection for gig workers, and see exactly where it fails.
>
> **One-line pitch:** RainProof is a simulation lab that shows how rainfall translates into modeled gig-worker income loss, how a parametric payout could offset that loss, and how much basis risk each protection design leaves behind.

---

## 📌 Framing & Purpose

### What RainProof IS:
* A quantitative simulation laboratory
* A data-science prototype for parametric risk design
* A basis risk stress-testing system
* A robustness evaluation engine across model uncertainty

### What RainProof IS NOT:
* ❌ An insurance company or broker
* ❌ An insurance quote engine or pricing tool
* ❌ A claim-processing system
* ❌ A real-world rider income predictor

---

## 📊 Observed vs. Modeled Data

Every visualization and metric clearly distinguishes factual observations from synthetic modeling assumptions:

* **🟢 OBSERVED:** Factual Open-Meteo historical daily weather for Mumbai (2019–2025), day of week, and Indian calendar dates.
* **🟠 MODELED:** Modeled baseline income (₹900/day), rain sensitivity parameter $s$, saturating rain stress, lognormal income noise, simulated payouts, and basis risk metrics.

---

## 📐 Mathematical Specification

### 1. Saturating Rain Stress
$$\text{RainStress} = 1 - \exp\left(-\frac{\text{rain\_mm}}{50}\right)$$

### 2. Modeled Income Simulator
$$\text{Income} = \text{Baseline} \times \text{DayFactor} \times \text{FestivalFactor} \times (1 - s \times \text{RainStress}) \times \text{Noise}$$
where $s \in \{0.25 \text{ (LOW)}, 0.40 \text{ (MEDIUM)}, 0.55 \text{ (HIGH)}\}$.

### 3. Baseline Loss Reference
$$\text{BaselineRef} = \text{Baseline} \times \text{DayFactor} \times \text{FestivalFactor}$$
$$\text{Loss} = \max(\text{BaselineRef} - \text{Income}, 0)$$

### 4. Parametric Payout Curve
$$\text{Payout} = \text{MaxPayout} \times \text{clip}\left(\frac{\text{rain} - \text{StartRain}}{\text{FullRain} - \text{StartRain}}, 0, 1\right)$$

### 5. Protection Metrics & Basis Risk
* **Loss Coverage:** $\frac{\sum \min(\text{Payout}_i, \text{Loss}_i)}{\sum \text{Loss}_i}$
* **Payout Precision:** $\frac{\sum \min(\text{Payout}_i, \text{Loss}_i)}{\sum \text{Payout}_i}$
* **Uncovered Loss:** $1 - \text{Coverage}$
* **Overpayment:** $1 - \text{Precision}$
* **Annual Premium:** $\text{Premium} = \frac{\text{ExpectedAnnualPayout}}{\text{TargetLossRatio}}$

### 6. Robust Score & Robust Design Selection
$$\text{RobustScore} = \min(\text{LossCoverage}, \text{PayoutPrecision}) \text{ across Low, Medium, and High sensitivities.}$$
Identifies the **"Most robust design under tested assumptions"** based on worst-case performance bounds.

---

## 🏗️ Architecture & Project Structure

```
rainproof/
├── app.py              # Main Streamlit UI entry point & navigation
├── data/
│   └── weather.csv     # Cached Mumbai historical weather dataset (2019-2025)
├── src/
│   ├── __init__.py
│   ├── weather.py      # Open-Meteo API integration & caching engine
│   ├── simulator.py    # Rain stress & income loss simulation engine
│   ├── payout.py       # Parametric payout curve & parameter validation
│   ├── metrics.py      # Coverage, precision, basis risk & premium calculations
│   ├── robustness.py   # Vectorized grid search & worst-case robustness scoring
│   └── forecast.py     # Monte Carlo forecast & explainability engine
├── tests/
│   ├── test_phase1.py  # Weather data engine validation
│   ├── test_simulator.py # Income simulation unit tests
│   ├── test_payout.py  # Parametric payout unit tests
│   ├── test_metrics.py # Metrics vector calculation unit tests
│   └── test_robustness.py # Robustness engine & execution speed tests
├── requirements.txt    # Package dependencies
└── README.md           # Documentation
```

---

## 🚀 Installation & Quickstart

```bash
# 1. Clone repository
git clone https://github.com/your-repo/RainProof.git
cd RainProof

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Run test suite
python -m pytest tests/

# 4. Launch Streamlit Application
streamlit run app.py
```

---

## 🗺️ Roadmap & Future Enhancements

- [ ] Spatial weather grids to address localized micro-climate basis risk.
- [ ] Integration of anonymized platform demand / order volume proxies.
- [ ] Multi-city expansion (Delhi, Bengaluru, Hyderabad, Chennai).
- [ ] Multi-peril protection models (Extreme heat index, flooding, air quality).

---

## ⚠️ Disclaimer

> **Prototype simulation. Not an insurance product or quote.**
> RainProof does not use proprietary rider earnings data. Income behavior is modeled using explicit assumptions. Results demonstrate how protection designs behave under those assumptions and should not be interpreted as predictions of actual individual earnings or insurance performance.
