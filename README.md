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
