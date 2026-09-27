import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="PRION-EARLY | Multisignal AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0c1d2f 48%, #102b3d 100%); }
    .block-container { max-width: 1200px; padding-top: 2rem; }
    .hero { padding: 2rem; border: 1px solid #24465b; border-radius: 24px;
            background: linear-gradient(120deg, rgba(11,39,57,.95), rgba(13,63,72,.75));
            margin-bottom: 1.5rem; }
    .hero h1 { color: #c9fff1; font-size: 3.2rem; margin: 0; letter-spacing: .04em; }
    .hero p { color: #a9c8d0; font-size: 1.1rem; max-width: 760px; }
    .card { background: rgba(18, 43, 58, .82); border: 1px solid #285269;
            border-radius: 18px; padding: 1.25rem; height: 100%; }
    .eyebrow { color: #5ce1c2; text-transform: uppercase; letter-spacing: .15em;
               font-size: .75rem; font-weight: 700; }
    .disclaimer { color: #ffd995; background: rgba(111,76,20,.28); border: 1px solid #8c6c2a;
                  border-radius: 12px; padding: .9rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Research prototype · multisignal intelligence</div>
      <h1>PRION-EARLY 🧬</h1>
      <p>Explore how multiple biological signals can be combined into an interpretable
      machine-learning pattern score.</p>
    </div>
    <div class="disclaimer">⚠️ <b>Educational proof of concept:</b> this site uses simulated data.
    It is not a medical diagnostic tool and must not be used for clinical decisions.</div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def build_demo_model():
    rng = np.random.default_rng(42)
    n = 500
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "RTQuIC": rng.normal(.35 + 1.15 * y, .25, n).clip(0),
        "NfL": rng.normal(180 + 650 * y, 90 + 80 * y, n).clip(1),
        "t-tau": rng.normal(450 + 2600 * y, 220 + 500 * y, n).clip(1),
        "14-3-3": rng.normal(.25 + .60 * y, .18, n).clip(0, 1),
        "PRNP variant": rng.binomial(1, .10 + .15 * y, n),
    })
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ai", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X, y)
    return model

model = build_demo_model()

with st.sidebar:
    st.markdown("## About this demo")
    st.write("PRION-EARLY demonstrates a reproducible workflow for combining research signals.")
    st.markdown("### Pipeline")
    st.write("Input signals → imputation → standardization → logistic regression → explanation")
    st.markdown("### Status")
    st.success("Prototype online")
    st.caption("Created by Noor Yasir")

st.markdown("## Analyze a multisignal pattern")
st.write("Enter illustrative values below, then run the model. Blank biological measurements can be represented with the available defaults in this demo.")

left, right = st.columns([1.15, 1], gap="large")
with left:
    with st.container(border=True):
        st.markdown("### 1 · Biological signals")
        c1, c2 = st.columns(2)
        with c1:
            rtquic = st.number_input("RT-QuIC / seeding signal", min_value=0.0, value=0.5, step=0.1, help="Synthetic relative signal.")
            nfl = st.number_input("NfL", min_value=0.0, value=200.0, step=10.0, help="Synthetic illustrative concentration.")
            tau = st.number_input("t-tau", min_value=0.0, value=500.0, step=50.0)
        with c2:
            marker = st.number_input("14-3-3 marker (normalized)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            variant = st.selectbox("PRNP variant in this demo", [0, 1], format_func=lambda x: "No / not detected" if x == 0 else "Yes / detected")
            st.caption("All values are simulated and have no clinical interpretation.")
        analyze = st.button("Analyze multisignal pattern", type="primary", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown("### 2 · AI output")
        st.info("Set the signals and select **Analyze multisignal pattern** to generate a demonstration score.")
        st.metric("Model-estimated pattern score", "—")

if analyze:
    X_input = pd.DataFrame([{"RTQuIC": rtquic, "NfL": nfl, "t-tau": tau, "14-3-3": marker, "PRNP variant": variant}])
    probability = float(model.predict_proba(X_input)[0, 1])
    if probability < .33:
        interpretation = "Lower pattern score in this demonstration model."
    elif probability < .67:
        interpretation = "Intermediate pattern score in this demonstration model."
    else:
        interpretation = "Higher pattern score in this demonstration model."

    st.markdown("### Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Pattern score", f"{probability:.1%}")
    r2.metric("Signals analyzed", "5")
    r3.metric("Model", "Logistic regression")
    st.info(interpretation)

    coefficients = model.named_steps["ai"].coef_[0]
    importance = pd.DataFrame({"Signal": X_input.columns, "Model coefficient": coefficients})
    importance["Absolute influence"] = importance["Model coefficient"].abs()
    importance = importance.sort_values("Absolute influence", ascending=False)
    chart_col, table_col = st.columns([1.2, 1])
    with chart_col:
        st.markdown("#### Relative signal influence")
        st.bar_chart(importance.set_index("Signal")["Absolute influence"], color="#5ce1c2")
    with table_col:
        st.markdown("#### Explanation data")
        st.dataframe(importance.round(3), hide_index=True, use_container_width=True)

    st.warning("This score was generated from simulated training data. It is not a clinical probability, diagnosis, prognosis, or validated prediction.")

st.markdown("---")
about, limits, next_steps = st.tabs(["What is the AI doing?", "Research limitations", "Next steps"])
with about:
    st.write("The model receives several numerical signals at once, imputes missing values, standardizes their scales, and learns statistical relationships between the combined feature pattern and simulated labels. The coefficient chart provides a simple global explanation of which inputs influenced the fitted model most strongly.")
with limits:
    st.write("Biomarker availability, laboratory methods, reference ranges, cohort composition, and outcome definitions vary across studies. A real research system would require ethically sourced data, predefined labels, nested cross-validation, calibration analysis, external validation, subgroup analysis, uncertainty estimates, and expert review.")
with next_steps:
    st.write("Replace the demonstration generator with a documented research dataset, add a reproducible training and evaluation pipeline, version the model and features, protect sensitive data, and validate performance prospectively before making any scientific claims.")

st.caption("PRION-EARLY · Multisignal AI research prototype · Synthetic data only")
