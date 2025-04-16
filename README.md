PCOS Diagnosis Predictor
A Machine Learning Web App for Polycystic Ovary Syndrome Risk Assessment

App Demo
(Replace with actual screenshot after deployment)

📌 Overview
This Streamlit-based web application predicts Polycystic Ovary Syndrome (PCOS) using a Random Forest Classifier trained on clinical biomarkers. Designed for healthcare professionals and researchers, it provides:
✔ Real-time risk probability
✔ Interactive data exploration
✔ Model explainability with SHAP values
✔ User-friendly interface with value range guidance

🚀 Quick Start
Prerequisites
Python 3.8+

Git

Installation
Clone the repository:

bash
Copy
git clone https://github.com/yourusername/pcos-predictor.git
cd pcos-predictor
Install dependencies:

bash
Copy
pip install -r requirements.txt
Download the dataset:

Place pcos_rotterdam_balanceado.csv in the project root

Run the App
bash
Copy
streamlit run app.py
Access the app at: http://localhost:8501

📂 Project Structure
Copy
pcos-predictor/
├── app.py               # Main Streamlit application
├── pcos_rotterdam_balanceado.csv  # Dataset
├── requirements.txt     # Python dependencies
└── README.md           # Documentation
🔧 Dependencies
streamlit

pandas

scikit-learn

matplotlib

seaborn

numpy

shap

📊 Features
Interactive Prediction: Input clinical values to get instant PCOS risk assessment

Data Exploration: Visualize dataset statistics and distributions

Model Insights: View feature importance and SHAP explanations

Responsive Design: Works on both desktop and mobile devices

📜 License
MIT License - Free for academic and clinical use

📸 Screenshots
(Add actual screenshots of your app here after deployment)

🌐 Live Demo
Streamlit Cloud (Add your deployment link here)

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
