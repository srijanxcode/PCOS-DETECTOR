# PCOS Diagnosis Predictor 🩺  
*A Machine Learning Web App for Polycystic Ovary Syndrome Risk Assessment*

![App Preview](https://via.placeholder.com/800x400/FF6B9E/FFFFFF?text=PCOS+Diagnosis+Predictor+Demo)

## Table of Contents
- [Features](#-features)
- [Local Installation](#-local-installation)
  - [Using Virtual Environment](#using-virtual-environment)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [License](#-license)

## ✨ Features
- **Real-time PCOS risk prediction** using Random Forest (AUC: 0.92)
- **Interactive data explorer** with statistical visualizations
- **Model explainability** via SHAP values
- **Range-guided input system** with clinical value references
- **Responsive design** for desktop and mobile use

## 💻 Local Installation

### Using Virtual Environment
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pcos-predictor.git
cd pcos-predictor

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset in project root
# Download pcos_rotterdam_balanceado.csv and place it here

# 5. Run the application
streamlit run app.py
