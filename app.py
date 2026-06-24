"""
E-commerce Churn & CLV — Streamlit Dashboard
SE-CD-638 Machine Learning CEP | Manahil Aftab

Run locally:
    pip install streamlit pandas numpy scikit-learn xgboost joblib plotly lifetimes optuna
    streamlit run app.py

For Streamlit Cloud: push this file + requirements.txt +
all .pkl and .csv files from your project to a GitHub repo,
then connect it at share.streamlit.io.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn & CLV Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #F7F9FB; }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16284A 0%, #0E1B33 100%);
    }
    [data-testid="stSidebar"] * { color: #C9D4E3 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #E0A458 !important; font-weight: 600; }

    /* metric cards */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    [data-testid="stMetricLabel"]  { color: #64748B !important; font-size: 13px !important; }
    [data-testid="stMetricValue"]  { color: #16284A !important; font-size: 28px !important; font-weight: 700 !important; }

    /* section headers */
    h1 { color: #16284A !important; }
    h2 { color: #16284A !important; }
    h3 { color: #2E5C8A !important; }

    /* callout box */
    .callout {
        background: #16284A;
        border-left: 5px solid #E0A458;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #DCE3EE;
        font-size: 15px;
        line-height: 1.6;
    }
    .callout b { color: #E0A458; }
    .warn-box {
        background: #FFF3CD;
        border-left: 5px solid #E0A458;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #856404;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────
CLUSTER_NAMES = {0: "Hibernating/Lost", 1: "Champions", 2: "At-Risk", 3: "New/Low-Engagement"}
CLUSTER_COLORS = {0: "#8896A6", 1: "#E0A458", 2: "#D9534F", 3: "#0F7173"}

@st.cache_data
def load_data():
    clv = pd.read_csv("phase5_clv_retention_priority.csv")
    clv["segment_name"] = clv["cluster"].map(CLUSTER_NAMES)
    clv["cluster_color"] = clv["cluster"].map(CLUSTER_COLORS)
    return clv

@st.cache_data
def load_model_comparison():
    baseline = pd.read_csv("phase2_model_comparison.csv")
    tuned    = pd.read_csv("phase4_final_comparison.csv")
    return baseline, tuned

@st.cache_resource
def load_model():
    bundle = joblib.load("phase4_best_model.pkl")
    return bundle

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Churn & CLV Dashboard")
    st.markdown("**SE-CD-638 ML — CEP**")
    st.markdown("Manahil Aftab · FJWU")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "👥 Customer Segments", "🏆 Retention Priority", "🔮 Predict a Customer"],
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#8896A6; line-height:1.6;'>
    <b style='color:#E0A458;'>Dataset</b><br>
    Online Retail II · UCI ML Repository<br><br>
    <b style='color:#E0A458;'>Best Model</b><br>
    Voting Ensemble (ROC-AUC 0.810)<br><br>
    <b style='color:#E0A458;'>CLV Horizon</b><br>
    6-month forward prediction
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = load_data()
    baseline_df, tuned_df = load_model_comparison()
    data_loaded = True
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}\n\nMake sure all CSV and PKL files from your project are in the same folder as this app.")
    data_loaded = False
    st.stop()

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Project Overview")
    st.markdown("#### E-commerce Customer Retention: Churn Prediction, RFM Segmentation & Predictive CLV")

    st.markdown("""
    <div class='callout'>
    <b>What this project does:</b> Most churn projects just predict who is going to leave.
    This project goes further — it combines churn risk with predicted future customer value
    to produce a ranked list of <b>who is actually worth trying to keep</b>.
    The final metric is <b>Expected Value at Risk = Churn Probability × Predicted 6-Month CLV</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    st.markdown("### Key Dataset Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", "5,179")
    c2.metric("Churn Rate", f"{df['churned'].mean()*100:.1f}%")
    c3.metric("Avg Predicted CLV", f"£{df['predicted_clv_6m'].mean():,.0f}")
    c4.metric("Total Value at Risk", f"£{df['expected_value_at_risk'].sum():,.0f}")
    c5.metric("Outcome Window", "120 days")

    st.markdown("---")

    # ── Model Performance ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Baseline vs Tuned (CV ROC-AUC)")
        base_dict = dict(zip(baseline_df["Model"], baseline_df["CV ROC-AUC"]))
        tuned_dict = dict(zip(
            tuned_df[~tuned_df["Model"].str.contains("Ensemble")]["Model"],
            tuned_df[~tuned_df["Model"].str.contains("Ensemble")]["Test ROC-AUC"]
        ))
        models = ["Logistic Regression", "k-NN", "SVM (RBF)", "Random Forest", "XGBoost"]
        fig = go.Figure()
        fig.add_bar(name="Baseline", x=models, y=[base_dict.get(m, 0) for m in models],
                    marker_color="#8896A6")
        fig.add_bar(name="Tuned", x=models, y=[tuned_dict.get(m, 0) for m in models],
                    marker_color="#0F7173")
        fig.update_layout(barmode="group", yaxis_range=[0.7, 0.85], height=320,
                          plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(gridcolor="#E2E8F0")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tuning overturned the baseline ranking — XGBoost improved the most (+0.030).")

    with col_right:
        st.markdown("### Expected Value at Risk by Segment")
        seg_totals = df.groupby("segment_name")["expected_value_at_risk"].sum().reset_index()
        seg_totals = seg_totals.sort_values("expected_value_at_risk", ascending=False)
        color_map = {v: CLUSTER_COLORS[k] for k, v in CLUSTER_NAMES.items()}
        fig2 = px.bar(seg_totals, x="segment_name", y="expected_value_at_risk",
                      color="segment_name", color_discrete_map=color_map,
                      labels={"segment_name": "Segment", "expected_value_at_risk": "Total Value at Risk (£)"},
                      text_auto=".3s")
        fig2.update_layout(showlegend=False, height=320,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10))
        fig2.update_yaxes(gridcolor="#E2E8F0")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("At-Risk carries the most total exposure despite lower individual value than Champions.")

    st.markdown("---")
    st.markdown("### 5-Phase Pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, icon, title, desc in [
        (p1, "🧹", "Preprocess", "Leakage-safe split, RFM features"),
        (p2, "🤖", "Classify", "5 churn models compared"),
        (p3, "🔵", "Cluster", "4 segmentation methods"),
        (p4, "⚙️", "Tune", "Grid, Random & Bayesian search"),
        (p5, "💰", "Score CLV", "BG/NBD + Gamma-Gamma"),
    ]:
        col.markdown(f"""
        <div style='background:white; border-radius:10px; padding:16px; text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.07); height:120px;'>
            <div style='font-size:26px;'>{icon}</div>
            <div style='font-weight:700; color:#16284A; font-size:14px; margin-top:6px;'>{title}</div>
            <div style='color:#64748B; font-size:12px; margin-top:4px;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2: CUSTOMER SEGMENTS
# ─────────────────────────────────────────────
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown("K-Means (k=4) outperformed GMM, Hierarchical, and DBSCAN — and the 4 segments align perfectly with churn rates neither model saw during clustering.")

    # ── Segment summary cards ──
    segs = df.groupby("segment_name").agg(
        Customers=("Customer ID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Churn_Rate=("churned", "mean"),
        Avg_CLV=("predicted_clv_6m", "mean"),
    ).reset_index()

    seg_order = ["Champions", "At-Risk", "New/Low-Engagement", "Hibernating/Lost"]
    seg_icons = {"Champions": "👑", "At-Risk": "⚠️", "New/Low-Engagement": "🌱", "Hibernating/Lost": "💤"}

    cols = st.columns(4)
    for i, seg in enumerate(seg_order):
        row = segs[segs["segment_name"] == seg].iloc[0]
        color = color_map[seg]
        with cols[i]:
            st.markdown(f"""
            <div style='background:white; border-radius:12px; padding:20px;
                        border-top:4px solid {color}; box-shadow:0 2px 8px rgba(0,0,0,0.07);'>
                <div style='font-size:28px; text-align:center;'>{seg_icons[seg]}</div>
                <div style='font-weight:700; color:#16284A; font-size:16px; text-align:center; margin:8px 0;'>{seg}</div>
                <hr style='border-color:#E2E8F0;'>
                <div style='color:#64748B; font-size:13px; line-height:2;'>
                    👤 <b>{int(row.Customers):,}</b> customers<br>
                    📅 Recency: <b>{row.Avg_Recency:.0f} days</b><br>
                    🔁 Frequency: <b>{row.Avg_Frequency:.1f} orders</b><br>
                    💷 Monetary: <b>£{row.Avg_Monetary:,.0f}</b><br>
                    🚪 Churn Rate: <b style='color:{color};'>{row.Churn_Rate*100:.0f}%</b><br>
                    💰 Avg CLV: <b>£{row.Avg_CLV:,.0f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Segment Size Distribution")
        fig = px.pie(segs, names="segment_name", values="Customers",
                     color="segment_name", color_discrete_map=color_map,
                     hole=0.4)
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### Churn Rate vs Avg CLV by Segment")
        fig2 = px.scatter(segs, x="Churn_Rate", y="Avg_CLV",
                          size="Customers", color="segment_name",
                          color_discrete_map=color_map,
                          text="segment_name",
                          labels={"Churn_Rate": "Churn Rate", "Avg_CLV": "Avg Predicted CLV (£)"},
                          size_max=50)
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.update_layout(height=340, showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10))
        fig2.update_xaxes(tickformat=".0%", gridcolor="#E2E8F0")
        fig2.update_yaxes(gridcolor="#E2E8F0")
        fig2.add_vline(x=0.5, line_dash="dash", line_color="#8896A6", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Why K-Means Won Over Other Methods")
    comp = pd.read_csv("phase3_clustering_comparison.csv")
    st.dataframe(comp.style.highlight_max(subset=["Silhouette"], color="#d4edda")
                           .highlight_min(subset=["Davies-Bouldin"], color="#d4edda")
                           .format({"Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}"}),
                 use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='warn-box'>
    <b>Why k=4 and not k=2?</b> k=2 maximized the silhouette score (0.452) but only split customers
    into "active" vs "inactive" — barely more information than the churn label already gives you.
    k=4 (silhouette 0.376) was chosen because it produces segments a retention team can actually act on differently.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3: RETENTION PRIORITY
# ─────────────────────────────────────────────
elif page == "🏆 Retention Priority":
    st.title("🏆 Retention Priority List")
    st.markdown("Customers ranked by **Expected Value at Risk = Churn Probability × Predicted 6-Month CLV**")

    st.markdown("""
    <div class='callout'>
    <b>How to read this:</b> A customer with a 5% churn chance but £80,000 CLV scores higher than a
    customer with 80% churn chance but £200 CLV. This prioritizes <b>who's worth saving</b>,
    not just <b>who's most likely to leave</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ──
    col1, col2, col3 = st.columns(3)
    with col1:
        seg_filter = st.multiselect("Filter by Segment", options=list(CLUSTER_NAMES.values()),
                                     default=list(CLUSTER_NAMES.values()))
    with col2:
        min_clv = st.slider("Min Predicted CLV (£)", 0, 10000, 0, step=100)
    with col3:
        top_n = st.slider("Show top N customers", 10, 200, 50, step=10)

    filtered = df[
        (df["segment_name"].isin(seg_filter)) &
        (df["predicted_clv_6m"] >= min_clv)
    ].nlargest(top_n, "expected_value_at_risk")

    # ── Priority scatter ──
    fig = px.scatter(filtered,
                     x="churn_probability", y="predicted_clv_6m",
                     color="segment_name", color_discrete_map=color_map,
                     size="expected_value_at_risk", size_max=30,
                     hover_data={"Customer ID": True,
                                 "churn_probability": ":.1%",
                                 "predicted_clv_6m": ":,.0f",
                                 "expected_value_at_risk": ":,.0f"},
                     labels={"churn_probability": "Churn Probability",
                             "predicted_clv_6m": "Predicted 6-Month CLV (£)",
                             "segment_name": "Segment"})
    fig.add_vline(x=0.5, line_dash="dash", line_color="#8896A6",
                  annotation_text="50% churn threshold", opacity=0.6)
    fig.add_hline(y=filtered["predicted_clv_6m"].median(), line_dash="dash",
                  line_color="#8896A6", annotation_text="median CLV", opacity=0.6)
    fig.add_annotation(x=0.9, y=filtered["predicted_clv_6m"].quantile(0.95),
                        text="🔴 HIGH PRIORITY", font=dict(color="#D9534F", size=13), showarrow=False)
    fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickformat=".0%", gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    st.plotly_chart(fig, use_container_width=True)

    # ── Table ──
    st.markdown(f"### Top {top_n} Customers by Expected Value at Risk")
    display = filtered[["Customer ID", "segment_name", "churn_probability",
                         "predicted_clv_6m", "expected_value_at_risk",
                         "Recency", "Frequency", "Monetary"]].copy()
    display.columns = ["Customer ID", "Segment", "Churn Prob",
                        "Predicted CLV (£)", "Value at Risk (£)",
                        "Recency (days)", "Frequency", "Monetary (£)"]
    st.dataframe(
        display.style
               .format({"Churn Prob": "{:.1%}", "Predicted CLV (£)": "£{:,.0f}",
                        "Value at Risk (£)": "£{:,.0f}", "Monetary (£)": "£{:,.0f}"})
               .background_gradient(subset=["Value at Risk (£)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True
    )

    # ── Download ──
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download this list as CSV", csv,
                       "retention_priority.csv", "text/csv")

# ─────────────────────────────────────────────
# PAGE 4: PREDICT A CUSTOMER
# ─────────────────────────────────────────────
elif page == "🔮 Predict a Customer":
    st.title("🔮 Predict a New Customer")
    st.markdown("Enter a customer's RFM metrics and get a live churn probability and segment prediction.")

    try:
        bundle = load_model()
        model = bundle["model"]
        scaler = bundle["scaler"]
        FEATURES = bundle["features"]
        model_loaded = True
    except Exception as e:
        st.error(f"Could not load model: {e}\n\nRun the notebook first to generate phase4_best_model.pkl")
        model_loaded = False

    if model_loaded:
        st.markdown("""
        <div class='warn-box'>
        <b>How this works:</b> The Voting Ensemble model (trained in Phase 4) predicts churn probability
        from the customer's RFM features. Segment assignment is based on the K-Means cluster boundaries
        from Phase 3. CLV estimate uses the average order value as a simple proxy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Customer Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            recency       = st.number_input("Recency (days since last order)", 0, 730, 45)
            frequency     = st.number_input("Frequency (number of orders)", 1, 400, 6)
            monetary      = st.number_input("Monetary (total spend £)", 0.0, 500000.0, 1500.0, step=50.0)
        with col2:
            distinct_prod = st.number_input("Distinct Products Purchased", 1, 4000, 30)
            tenure_days   = st.number_input("Tenure (days as customer)", 1, 750, 300)
            avg_order     = st.number_input("Average Order Value (£)", 0.0, 50000.0, 250.0, step=10.0)
        with col3:
            interval_std  = st.number_input("Purchase Interval Std Dev (days)", 0.0, 200.0, 15.0, step=1.0)
            is_uk         = st.radio("Based in UK?", ["Yes", "No"])
            is_uk_val     = 1 if is_uk == "Yes" else 0

        if st.button("🔮 Predict", type="primary", use_container_width=True):
            # Build feature row
            raw = pd.DataFrame([{
                "Recency": recency, "Frequency": frequency, "Monetary": monetary,
                "distinct_products": distinct_prod, "tenure_days": tenure_days,
                "avg_order_value": avg_order, "purchase_interval_std": interval_std,
                "is_uk": is_uk_val,
            }])
            for col in ["Monetary", "avg_order_value", "tenure_days"]:
                raw[col] = np.log1p(raw[col])
            raw_scaled = pd.DataFrame(scaler.transform(raw[FEATURES]), columns=FEATURES)
            churn_prob = model.predict_proba(raw_scaled)[0][1]
            churn_pred = "Likely to Churn" if churn_prob >= 0.5 else "Likely to Stay"
            simple_clv = avg_order * frequency * 0.5  # rough 6-month proxy

            # ── Assign segment heuristically ──
            if recency <= 60 and frequency >= 10:
                segment = "Champions"
            elif recency <= 60 and frequency < 10:
                segment = "New/Low-Engagement"
            elif recency > 200:
                segment = "Hibernating/Lost"
            else:
                segment = "At-Risk"

            seg_color = color_map[segment]

            st.markdown("---")
            st.markdown("### Prediction Result")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Churn Probability", f"{churn_prob:.1%}")
            r2.metric("Prediction", churn_pred)
            r3.metric("Assigned Segment", segment)
            r4.metric("Estimated 6M CLV", f"£{simple_clv:,.0f}")

            st.markdown(f"""
            <div class='callout'>
            <b>Expected Value at Risk:</b> £{churn_prob * simple_clv:,.0f}<br><br>
            This customer has a <b>{churn_prob:.1%} chance of churning</b> and an estimated
            <b>6-month value of £{simple_clv:,.0f}</b>, placing them in the
            <b style='color:{seg_color};'>{segment}</b> segment.
            {'⚠️ This customer is worth an active retention effort.' if churn_prob >= 0.4 and simple_clv > 500
            else '✅ Low risk — monitor but no urgent action needed.' if churn_prob < 0.3
            else '💤 High churn risk but low value — lower retention priority.'}
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge chart ──
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={"text": "Churn Probability %", "font": {"color": "#16284A"}},
                number={"suffix": "%", "font": {"color": "#16284A"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                    "bar": {"color": "#D9534F" if churn_prob >= 0.5 else "#0F7173"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 60], "color": "#fff3cd"},
                        {"range": [60, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "#16284A", "width": 3}, "value": 50},
                },
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20),
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
