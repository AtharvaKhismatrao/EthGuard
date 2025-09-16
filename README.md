# EthGuard

A machine learning-powered web application that evaluates the fraud risk of Ethereum addresses by analyzing transaction behavior, network patterns, and blockchain interactions. EthGuard provides a comprehensive fraud detection framework for cryptocurrency transaction analysis.

---

## Features

- **Real-time Risk Assessment**: Immediate scoring of Ethereum addresses with color-coded risk indicators
- **Advanced Machine Learning**: LightGBM model trained on combined Etherscan and Kaggle datasets
- **Multi-faceted Analysis**:
  - Transaction pattern detection
  - Network connection mapping
  - Temporal behavior analysis
  - Token interaction evaluation
  - Compliance and sanctions screening
- **Sophisticated UI Dashboard**: Interactive interface with multiple analysis tabs
- **Fallback Assessment System**: Pattern-based risk evaluation when ML model cannot be used
- **Pre-trained Model**: Includes `lightgbm_fraud_model.pkl` for instant predictions without training
- **Model Ensemble Architecture**: Tested with Random Forest, XGBoost, and LightGBM for optimal performance

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

## Images

![Alt text](path/to/Screenshot (2).png)


![Alt text](path/to/Screenshot (3).png)
## User Interface

The application features a modern, responsive dashboard with six analytical tabs:

- **Overview**: General address information and risk summary
- **Risk Analysis**: Detailed breakdown of fraud indicators
- **Network**: Visualization of incoming and outgoing connections
- **Patterns**: Transaction pattern detection and anomaly highlighting
- **Compliance**: Sanctions screening and regulatory flag checking
- **History**: Temporal analysis of transaction behavior

Each section provides specialized metrics and visualizations to support comprehensive fraud detection.

---

## How it Works
**Workflow:**
1. **User Input**: Enter an Ethereum address in the analysis dashboard
2. **Backend Processing**:
   - Pre-trained LightGBM model evaluates address features
   - When address is unknown to the model, fallback pattern-based assessment activates
   - Features are extracted (transaction patterns, network stats, temporal indicators)
   - Multiple risk factors are calculated and aggregated
3. **Frontend Output**:
   - Interactive dashboard displays comprehensive risk analysis
   - Risk probability percentage with color-coded risk level
   - Detailed metrics across multiple analysis categories
   - Network visualizations and pattern detection results

---

## Risk Levels
- 🟢 **Low/No Risk** → Fraud Probability < 40%
- 🟡 **Medium Risk** → Fraud Probability 40–70%
- 🔴 **High Risk** → Fraud Probability > 70%

---

## Technical Architecture
- **Data Processing**: Features are extracted from transaction history and network behavior
- **Model Pipeline**: Address data → Feature extraction → Risk prediction → Detailed analysis
- **Fallback System**: Pattern-based assessment when ML prediction isn't possible
- **Visualization Engine**: Converts complex metrics into intuitive visual indicators

---

## Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: LightGBM, Scikit-learn, XGBoost
- **Data Sources**: Kaggle Ethereum Fraud Detection Dataset, Etherscan API
- **Visualization**: Custom CSS with responsive design elements
- **Model Storage**: Pre-trained model in pickle format (`lightgbm_fraud_model.pkl`)

---

## Future Improvements
- Integrate live data from Ethereum blockchain using Web3.py
- Add Metamask wallet connection for real-time address checks
- Implement WebSocket for live transaction monitoring
- Expand the model to support other EVM-compatible blockchains
- Develop API endpoints for integration with other services
- Create user accounts with saved address watchlists
- Deploy as a cloud-based API (AWS / Azure / GCP)
- Add comparative analysis between multiple addresses

---

## License
[MIT License](LICENSE)
