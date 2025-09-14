# 🛍️ Retail Demand Forecasting and Price Optimization

A machine learning project to forecast retail store sales and simulate the impact of price changes on demand.

---

## 📌 Project Goals

- Forecast item-level sales using real historical data  
- Simulate sales impact of pricing changes using a trained XGBoost model  
- Provide actionable insights for price optimization  

---

## 🧰 Tools & Technologies

- **Python** (Pandas, XGBoost, Scikit-Learn)  
- **Streamlit** for interactive dashboards  
- **SHAP** for model explainability  
- **Git, GitHub** for version control  

---

## 💡 Key Features

- Cleaned and joined sales and store data  
- Feature engineering for holidays, promotions, seasonality  
- XGBoost regression model with SHAP explainability  
- Synthetic price simulation to assess impact  
- Streamlit dashboard with:  
  - Forecasting tab  
  - Price simulation slider  
  - SHAP visualizations  

---

## 📂 Project Structure

```bash
retail_demand_forecasting/
├── data/                            # Local folder, ignored via .gitignore
│   └── features_rossmann.csv        # Final training data (not in repo)
├── outputs/
│   └── price_simulation_predictions.csv   # Simulation results (local only)
├── notebooks/
│   ├── eda_notebook.ipynb           # EDA and preprocessing
│   └── price_simulation.ipynb       # Price simulation notebook
├── dashboards/
│   └── streamlit_app.py             # Interactive dashboard
├── models/
│   └── xgb_model.pkl                # Trained model (if needed)
├── utils/
│   └── feature_engineering.py       # Feature engineering script
├── README.md
└── .gitignore
🚫 Excluded from Git
The following files are large and tracked only locally:

bash
Copy code
data/features_rossmann.csv  
outputs/price_simulation_predictions.csv  
To avoid issues with GitHub file size limits, these are excluded via .gitignore.

🖥️ How to Run
Clone the repo

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the dashboard:

bash
Copy code
streamlit run dashboards/streamlit_app.py
📊 Sample Outputs (Optional)
Price simulation results using SHAP values

Streamlit dashboard view (add screenshot here later)

Predicted sales vs. actual sales plot

🔒 License
MIT License

👩‍💻 Author
Arushi Sharma
