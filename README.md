# EthGuard

A machine learning-powered web application that evaluates the fraud risk of Ethereum addresses by analyzing their transaction behavior and network patterns.


---


## Features

- Real-time risk scoring of Ethereum addresses
- Machine Learning (LightGBM) trained on Etherscan + Kaggle datasets
- Interactive web interface (Flask + HTML/CSS/JS frontend)
- Stacked Ensemble Models (tested Random Forest, XGBoost, LightGBM)
- Pre-trained model (lightgbm_fraud_model.pkl) included for instant predictions

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ethereum-fraud-detector.git
cd ethereum-fraud-detector
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```
4. Open your browser at http://localhost:5000

---

## How it Works
**Workflow:**
  1. User Input: Enter an Ethereum address.
  2. Backend Processing:
    - Pre-trained LightGBM model is loaded.
    - Features are extracted (transaction patterns, address stats, historical indicators).
    - Risk probability is generated.
  3. Frontend Output:
    - Displays fraud probability %.
    - Categorizes into Low, Medium, High Risk.
---
## Risk Levels
- 🟢 Low/No Risk → Fraud Probability < 40%
- 🟡 Medium Risk → Fraud Probability 40–70%
- 🔴 High Risk → Fraud Probability > 70%
---
## Tech Stack
- Backend: Flask (Python).
- Frontend: HTML, CSS, JavaScript.
- Machine Learning: LightGBM, Scikit-learn, Keras, XGBoost.
- Data: Kaggle Ethereum Fraud Detection Dataset + Etherscan API.
- Model Storage: lightgbm_fraud_model.pkl (pre-trained model).
---
## Future Improvements
- Integrate live data from Ethereum blockchain (Web3.py).
- Add Metamask wallet connection for real-time address checks.
- Deploy as a cloud-based API (AWS / Azure / GCP).
