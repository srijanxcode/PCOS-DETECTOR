import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import warnings
warnings.filterwarnings("ignore")

# Custom color theme
PINK_PURPLE_THEME = {
    "primaryColor": "#FF6B9E",
    "backgroundColor": "#F8E8FF",
    "secondaryBackgroundColor": "#F2D1FF",
    "textColor": "#4B0082",
    "font": "sans serif"
}

# Apply theme
st.set_page_config(
    page_title="PCOS Diagnosis Predictor", 
    layout="wide",
    page_icon="🌸"
)

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {PINK_PURPLE_THEME['backgroundColor']};
    }}
    .css-1d391kg {{
        background-color: {PINK_PURPLE_THEME['secondaryBackgroundColor']};
    }}
    h1, h2, h3 {{
        color: {PINK_PURPLE_THEME['textColor']};
    }}
    .st-b7 {{
        color: {PINK_PURPLE_THEME['textColor']};
    }}
    .st-cb {{
        background-color: {PINK_PURPLE_THEME['primaryColor']};
    }}
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pcos_rotterdam_balanceado.csv")
        if 'PCOS_Diagnosis' not in df.columns:
            st.error("Dataset must contain 'PCOS_Diagnosis' column")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_model(_df):
    if _df is None:
        return None
    X = _df.drop("PCOS_Diagnosis", axis=1)
    y = _df["PCOS_Diagnosis"]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X.columns

def main():
    st.title("🌸 PCOS Diagnosis Predictor")
    
    df = load_data()
    if df is None:
        st.stop()
    
    model, feature_columns = load_model(df)
    
    # Sidebar navigation
    st.sidebar.title("🌸 Navigation")
    app_mode = st.sidebar.radio(
        "Choose a page",
        ["🏠 Home", "📊 Data Exploration", "📈 Model Evaluation", "🔮 PCOS Prediction"],
        index=0
    )
    
    if app_mode == "🏠 Home":
        st.header("Understanding Polycystic Ovary Syndrome")
        st.markdown("""
        <div style='background-color:#F2D1FF; padding:20px; border-radius:10px;'>
        <h3 style='color:#4B0082;'>What is PCOS?</h3>
        <p>Polycystic ovary syndrome (PCOS) is a hormonal disorder common among women of reproductive age. 
        Women with PCOS may have infrequent or prolonged menstrual periods or excess male hormone (androgen) levels.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color:#F2D1FF; padding:20px; border-radius:10px; margin-top:20px;'>
        <h3 style='color:#4B0082;'>Common Symptoms</h3>
        <ul>
            <li>Irregular periods</li>
            <li>Excess hair growth</li>
            <li>Acne</li>
            <li>Weight gain</li>
            <li>Thinning hair</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif app_mode == "📊 Data Exploration":
        st.header("Exploring PCOS Data")
        
        st.subheader("Dataset Overview")
        st.write(f"Dataset shape: {df.shape}")
        
        if st.checkbox("Show raw data"):
            st.dataframe(df.style.applymap(lambda x: f"background-color: {PINK_PURPLE_THEME['secondaryBackgroundColor']}"))
        
        st.subheader("PCOS Diagnosis Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(df["PCOS_Diagnosis"].value_counts(), 
               labels=["No PCOS", "PCOS"], 
               colors=["#FF9EDF", "#BA68C8"],
               autopct='%1.1f%%')
        ax.set_title("Diagnosis Distribution")
        st.pyplot(fig)
        
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title("Feature Correlation Matrix")
        st.pyplot(fig)
        
    elif app_mode == "📈 Model Evaluation":
        st.header("Model Performance Analysis")
        
        X = df.drop("PCOS_Diagnosis", axis=1)
        y = df["PCOS_Diagnosis"]
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=["#FF9EFF", "#D291FF"], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())
        
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#BA68C8", label=f"ROC curve (AUC = {roc_auc_score(y, y_prob):.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color="#FF6B9E")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
        
    elif app_mode == "🔮 PCOS Prediction":
        st.header("Personalized PCOS Risk Assessment")
        st.markdown("""
        <div style='background-color:#F2D1FF; padding:20px; border-radius:10px;'>
        <h3 style='color:#4B0082;'>Instructions</h3>
        <p>Please enter the patient's clinical and biochemical features below. 
        The system will analyze the values and provide a risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(feature_columns):
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            median_val = float(df[col].median())
            
            range_info = f"[Min: {min_val:.1f}, Max: {max_val:.1f}]"
            
            if i % 2 == 0:
                with col1:
                    value = st.number_input(
                        f"**{col}** {range_info}",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=0.1,
                        help=f"Enter value between {min_val:.1f} and {max_val:.1f}"
                    )
            else:
                with col2:
                    value = st.number_input(
                        f"**{col}** {range_info}",
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        step=0.1,
                        help=f"Enter value between {min_val:.1f} and {max_val:.1f}"
                    )
            
            input_data[col] = value
        
        if st.button("🔍 Analyze PCOS Risk", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.markdown(f"""
                <div style='background-color:#FFD6E7; padding:20px; border-radius:10px; border-left: 6px solid #FF6B9E;'>
                <h3 style='color:#D1006A;'>⚠️ PCOS Risk Detected</h3>
                <p>Probability: <span style='font-weight:bold; color:#D1006A;'>{probability:.1%}</span></p>
                <p>Recommend consulting with an endocrinologist or gynecologist for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color:#E6F9FF; padding:20px; border-radius:10px; border-left: 6px solid #4B0082;'>
                <h3 style='color:#4B0082;'>✅ Low PCOS Risk</h3>
                <p>Probability: <span style='font-weight:bold; color:#4B0082;'>{probability:.1%}</span></p>
                <p>No significant indicators of PCOS detected based on the provided data.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='margin-top:20px;'>
            <h4>PCOS Risk Level</h4>
            <progress value="{probability}" max="1" style="width:100%; height:30px; accent-color:{PINK_PURPLE_THEME['primaryColor']};"></progress>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()