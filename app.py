import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="FAILSAFE", page_icon="🎓", layout="wide")

@st.cache_resource
def load_models():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    feature_cols = pd.read_csv('models/feature_columns.csv').columns.tolist()
    return model, explainer, feature_cols

model, explainer, feature_cols = load_models()

def generate_intervention(top_features):
    rules = {
        'absences':   'Schedule attendance counselling immediately.',
        'studytime':  'Recommend structured study plan (min 2hrs/day).',
        'failures':   'Assign peer tutor for weak subjects.',
        'Dalc':       'Refer to student wellness counsellor.',
        'Walc':       'Refer to student wellness counsellor.',
        'goout':      'Advise better time management.',
        'freetime':   'Suggest joining academic clubs.',
        'health':     'Recommend campus health centre visit.',
        'famrel':     'Connect with student support services.',
        'internet':   'Ensure library/lab internet access.',
    }
    interventions = []
    for f in top_features:
        if f in rules and len(interventions) < 3:
            interventions.append(f"• **{f.upper()}**: {rules[f]}")
    if not interventions:
        interventions.append("• General academic support recommended.")
    return '\n'.join(interventions)

# ── UI ────────────────────────────────────────────────────────────
st.title("🎓 FAILSAFE")
st.caption("Early student failure detection powered by XGBoost + SHAP")

tab1, tab2 = st.tabs(["Bulk Upload", "Risk Dashboard"])

with tab1:
    st.subheader("Upload student data CSV")
    st.caption("CSV must contain these columns: " + ", ".join(feature_cols[:8]) + "...")
    
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        # Handle student_id column if present
        id_col = None
        if 'student_id' in df.columns:
            id_col = df['student_id']
            df = df.drop(columns=['student_id'])
        
        # Keep only known feature columns
        available = [c for c in feature_cols if c in df.columns]
        df = df[available]
        
        # Predict
        risk_scores = model.predict_proba(df)[:, 1]
        risk_labels = ['HIGH RISK' if s > 0.5 else 'LOW RISK' for s in risk_scores]
        
        results = df.copy()
        if id_col is not None:
            results.insert(0, 'student_id', id_col.values)
        results['risk_score'] = np.round(risk_scores, 3)
        results['risk_label'] = risk_labels
        
        # Show summary
        high_risk = sum(1 for l in risk_labels if l == 'HIGH RISK')
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", len(df))
        col2.metric("High Risk", high_risk, delta=f"{high_risk/len(df)*100:.0f}%", delta_color="inverse")
        col3.metric("Low Risk", len(df) - high_risk)
        
        # Show table sorted by risk
        st.subheader("Risk Rankings")
        display_cols = ['student_id'] if id_col is not None else []
        display_cols += ['risk_score', 'risk_label', 'absences', 'studytime', 'failures']
        display_cols = [c for c in display_cols if c in results.columns]
        
        st.dataframe(
            results[display_cols].sort_values('risk_score', ascending=False),
            use_container_width=True
        )
        
        # Store for dashboard tab
        st.session_state['results'] = results
        st.session_state['df'] = df

with tab2:
    st.subheader("Individual Student Analysis")
    
    if 'results' not in st.session_state:
        st.info("Upload a CSV in the Bulk Upload tab first.")
    else:
        results = st.session_state['results']
        df = st.session_state['df']
        
        # Select student
        if 'student_id' in results.columns:
            student_options = results['student_id'].tolist()
        else:
            student_options = [f"Student {i}" for i in range(len(results))]
        
        selected = st.selectbox("Select a student to analyse", student_options)
        idx = student_options.index(selected)
        
        student_row = df.iloc[idx:idx+1]
        risk_score = results.iloc[idx]['risk_score']
        risk_label = results.iloc[idx]['risk_label']
        
        # Risk score display
        col1, col2 = st.columns(2)
        col1.metric("Risk Score", f"{risk_score:.2f}/1.00")
        col2.metric(
            "Status",
            risk_label,
            delta="Needs intervention" if risk_label == "HIGH RISK" else "Monitoring",
            delta_color="inverse" if risk_label == "HIGH RISK" else "normal"
        )
        st.progress(float(risk_score))
        
        # SHAP explanation
        sv = explainer.shap_values(student_row)[0]
        shap_df = pd.DataFrame({
            'Feature': feature_cols[:len(sv)],
            'SHAP Value': sv
        }).sort_values('SHAP Value', key=abs, ascending=False).head(5)
        
        st.subheader("Why is this student at risk?")
        for _, row in shap_df.iterrows():
            direction = "↑ increases risk" if row['SHAP Value'] > 0 else "↓ decreases risk"
            st.write(f"**{row['Feature']}**: {direction} (impact: {abs(row['SHAP Value']):.3f})")
        
        # Intervention plan
        st.subheader("Recommended Interventions")
        top_features = shap_df['Feature'].tolist()
        plan = generate_intervention(top_features)
        st.markdown(plan)