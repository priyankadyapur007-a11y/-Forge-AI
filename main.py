from __future__ import annotations

import io
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(
    page_title="FORGE AI · Data Science Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eef2ff; --muted:#9ba7c4; --panel:#111a2d; --line:rgba(173,190,230,.13); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 10% 0%, rgba(89,83,190,.19), transparent 34%),
    radial-gradient(ellipse at 95% 12%, rgba(22,174,169,.11), transparent 30%),
    #080d19;
  color:var(--ink);
}
[data-testid="stSidebar"] { background:#0b1221; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1450px; }
h1,h2,h3 { letter-spacing:-.045em; color:#f4f6ff; }
p,li,label { color:#c3cce0; }
.eyebrow { font:500 11px 'DM Mono',monospace; letter-spacing:.18em; color:#8d9bbd; text-transform:uppercase; }
.hero {
  padding:30px 32px; border:1px solid rgba(160,177,226,.18); border-radius:22px;
  background:linear-gradient(120deg,rgba(35,48,83,.82),rgba(16,28,45,.72));
  box-shadow:0 20px 70px rgba(0,0,0,.18); position:relative; overflow:hidden;
}
.hero:after { content:""; position:absolute; width:260px;height:260px;border:1px solid rgba(128,230,218,.16);
  border-radius:50%; right:-80px;top:-120px; box-shadow:0 0 0 24px rgba(128,230,218,.025),0 0 0 52px rgba(128,230,218,.02); }
.hero h1 { font-size:clamp(2.3rem,5vw,4.1rem); margin:8px 0 4px; }
.hero p { max-width:700px; color:#b5c1dc; font-size:1rem; }
.tag { display:inline-block; border:1px solid rgba(126,224,213,.28); color:#83e0d5;
  border-radius:99px; padding:5px 10px; font:11px 'DM Mono',monospace; margin-right:6px; }
.panel { background:rgba(17,26,45,.78); border:1px solid var(--line); border-radius:17px; padding:18px 20px; }
.metric { background:linear-gradient(145deg,rgba(28,42,69,.9),rgba(16,25,43,.9));
  border:1px solid var(--line); border-radius:15px; padding:16px 18px; min-height:104px; }
.metric .label { color:#95a4c4; font:11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
.metric .value { font-size:1.8rem; font-weight:800; color:#f4f6ff; margin-top:6px; }
.small-note { color:#8795b4; font-size:.84rem; }
div[data-testid="stMetric"] { background:rgba(20,31,52,.8); border:1px solid var(--line); padding:13px 16px; border-radius:14px; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(130,220,211,.3); background:linear-gradient(100deg,#315d70,#454a88); color:white; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#8ce3d8; color:white; }
.stTabs [data-baseweb="tab"] { color:#aebbd5; }
hr { border-color:var(--line); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
code { color:#9fe7dc !important; }
</style>
""", unsafe_allow_html=True)

for key, default in {
    "df": None, "filename": None, "target": None, "task": "Auto-detect",
    "models": {}, "best_name": None, "best_pipeline": None,
    "X_test": None, "y_test": None, "predictions": None,
    "feature_names": None, "run_at": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def card_metric(label, value, detail=""):
    st.markdown(
        f'<div class="metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="small-note">{detail}</div></div>',
        unsafe_allow_html=True
    )

def infer_task(series: pd.Series) -> str:
    if series.dtype == "object" or str(series.dtype) == "category" or series.dtype == bool:
        return "Classification"
    unique = series.nunique(dropna=True)
    if unique <= 12:
        return "Classification"
    return "Regression"

def make_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

def model_candidates(task):
    if task == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=180, random_state=42, class_weight="balanced", n_jobs=-1),
        }
    return {
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=180, random_state=42, n_jobs=-1),
    }

with st.sidebar:
    st.markdown('<div class="eyebrow">✦ FORGE / LAB 01</div>', unsafe_allow_html=True)
    st.markdown("## Your data, decoded.")
    st.caption("A compact workspace for exploring data and testing machine-learning ideas.")
    page = st.radio(
        "WORKSPACE",
        ["Mission Control", "Data Explorer", "Visual Studio", "Model Forge", "Model Report"],
        label_visibility="visible",
    )
    st.divider()
    uploaded = st.file_uploader("Load a CSV dataset", type=["csv"], help="Your file is processed in this app session.")
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.session_state.df = new_df
            st.session_state.models = {}
            st.session_state.best_pipeline = None
           
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")
    if st.session_state.df is not None:
        st.markdown("**ACTIVE DATASET**")
        st.write(f"`{st.session_state.filename}`")
        st.caption(f"{len(st.session_state.df):,} rows · {st.session_state.df.shape[1]} columns")
        if st.button("Clear dataset", use_container_width=True):
            for k in ["df","filename","target","models","best_name","best_pipeline","X_test","y_test","predictions"]:
                st.session_state[k] = None if k not in ["models"] else {}
            st.rerun()
    st.divider()
    st.markdown('<div class="small-note">BUILT FOR CURIOUS MINDS · VERSION 1.0</div>', unsafe_allow_html=True)

df = st.session_state.df

if page == "Mission Control":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">INTELLIGENT DATA SCIENCE LABORATORY</div>
      <h1>FORGE<span style="color:#80dfd3">.</span>AI</h1>
      <p>Turn a raw spreadsheet into a clearer story. Explore the shape of your data, build a few baseline models, and see what the numbers are really saying.</p>
      <span class="tag">DATA EXPLORATION</span><span class="tag">AUTOML BASELINES</span><span class="tag">MODEL INSIGHTS</span>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    if df is None:
        left, right = st.columns([1.35,1])
        with left:
            st.markdown("### Start with a dataset")
            st.write("Drop a CSV into the uploader in the sidebar. FORGE will give you a quick read on its structure before you decide what to do next.")
            st.markdown('<div class="panel"><div class="eyebrow">A GOOD FIRST TEST</div><h3 style="margin:8px 0">Iris flower dataset</h3><p>Small, tidy, and useful for a first classification experiment. Try predicting the flower species from its measurements.</p></div>', unsafe_allow_html=True)
            st.markdown("**Quick start**")
            st.markdown("1. Upload a CSV from the sidebar.  \n2. Open **Data Explorer** to inspect columns and missing values.  \n3. Choose a target column in **Model Forge**.  \n4. Compare baseline models and review the report.")
        with right:
            st.markdown('<div class="panel"><div class="eyebrow">THE WORKFLOW</div><h3>From rows to reasoning</h3><p>01 · Inspect the dataset</p><p>02 · Find patterns visually</p><p>03 · Train baseline models</p><p>04 · Compare metrics</p><p>05 · Export your experiment</p><hr><p class="small-note">Tip: start with a dataset that has a clear column you want to predict.</p></div>', unsafe_allow_html=True)
    else:
        missing = int(df.isna().sum().sum())
        dupes = int(df.duplicated().sum())
        a,b,c,d = st.columns(4)
        with a: card_metric("Rows", f"{len(df):,}", "observations")
        with b: card_metric("Features", f"{df.shape[1]:,}", "columns")
        with c: card_metric("Missing cells", f"{missing:,}", "values to review")
        with d: card_metric("Duplicate rows", f"{dupes:,}", "exact matches")
        st.write("")
        st.markdown("### A first look")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        st.caption("This is a preview, not a verdict. Check column meanings and data quality before training.")
        st.markdown("### Dataset fingerprint")
        types = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        fig = px.pie(types, names="Data type", values="Columns", hole=.62, title="Column types")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Explorer":
    st.markdown('<div class="eyebrow">01 / UNDERSTAND THE INPUT</div>', unsafe_allow_html=True)
    st.title("Data Explorer")
    if df is None:
        st.info("Upload a CSV in the sidebar to open your dataset.")
    else:
        st.markdown(f"**{st.session_state.filename}** · {len(df):,} rows × {df.shape[1]} columns")
        tab1, tab2, tab3 = st.tabs(["Preview", "Column profile", "Summary"])
        with tab1:
            st.dataframe(df, use_container_width=True, height=420)
        with tab2:
            profile = pd.DataFrame({
                "Column": df.columns,
                "Type": [str(x) for x in df.dtypes],
                "Missing": df.isna().sum().values,
                "Unique values": df.nunique(dropna=True).values,
                "Missing %": (df.isna().mean().values * 100).round(1),
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)
            st.caption("A high number of unique values is not automatically a problem; it depends on what the column represents.")
        with tab3:
            st.dataframe(df.describe(include="all").T, use_container_width=True)

elif page == "Visual Studio":
    st.markdown('<div class="eyebrow">02 / FIND THE SHAPE OF THINGS</div>', unsafe_allow_html=True)
    st.title("Visual Studio")
    if df is None:
        st.info("Upload a CSV in the sidebar to build visualisations.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()
        kind = st.selectbox("Choose a visual", ["Distribution", "Scatter plot", "Box plot", "Correlation map", "Category counts"])
        if kind == "Distribution":
            col = st.selectbox("Column", numeric_cols or all_cols)
            fig = px.histogram(df, x=col, marginal="box", title=f"Distribution · {col}", color_discrete_sequence=["#80dfd3"])
        elif kind == "Scatter plot":
            if len(numeric_cols) < 2:
                st.warning("A scatter plot needs at least two numeric columns.")
                fig = None
            else:
                x = st.selectbox("Horizontal axis", numeric_cols, index=0)
                y = st.selectbox("Vertical axis", numeric_cols, index=min(1,len(numeric_cols)-1))
                color = st.selectbox("Colour by (optional)", ["None"] + all_cols)
                fig = px.scatter(df, x=x, y=y, color=None if color=="None" else color, title=f"{y} vs {x}", opacity=.78)
        elif kind == "Box plot":
            col = st.selectbox("Numeric column", numeric_cols or all_cols)
            group = st.selectbox("Group by (optional)", ["None"] + all_cols)
            fig = px.box(df, x=None if group=="None" else group, y=col, points="outliers", title=f"Spread · {col}")
        elif kind == "Correlation map":
            if len(numeric_cols) < 2:
                st.warning("A correlation map needs at least two numeric columns.")
                fig = None
            else:
                corr = df[numeric_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Tealgrn", zmin=-1, zmax=1, title="Numeric feature correlations")
        else:
            col = st.selectbox("Category column", all_cols)
            counts = df[col].astype(str).value_counts().head(25).reset_index()
            counts.columns = [col, "Count"]
            fig = px.bar(counts, x=col, y="Count", title=f"Most common values · {col}", color="Count", color_continuous_scale="Teal")
        if fig is not None:
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce5fa")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Charts help you notice patterns. They do not prove that one variable causes another.")

elif page == "Model Forge":
    st.markdown('<div class="eyebrow">03 / TRAIN · COMPARE · LEARN</div>', unsafe_allow_html=True)
    st.title("Model Forge")
    st.write("Run a small, reproducible baseline comparison. FORGE handles missing values and categorical columns inside each model pipeline.")
    if df is None:
        st.info("Upload a CSV in the sidebar before starting an experiment.")
    else:
        eligible = [c for c in df.columns if df[c].notna().any()]
        if len(eligible) < 2:
            st.warning("You need at least two usable columns: one or more features and one target.")
        else:
            with st.container(border=True):
                target = st.selectbox("What should the model predict?", eligible, index=(eligible.index(st.session_state.target) if st.session_state.target in eligible else len(eligible)-1))
                suggested = infer_task(df[target].dropna())
                task = st.radio("Problem type", ["Classification", "Regression"], horizontal=True, index=0 if suggested=="Classification" else 1)
                test_size = st.slider("Test data (%)", 15, 40, 20, 5)
                st.caption(f"Suggested from the target column: {suggested}. Change it if you know the problem type.")
                run = st.button("✦ Run model comparison", type="primary", use_container_width=True)
            if run:
                work = df.dropna(subset=[target]).copy()
                X = work.drop(columns=[target])
                y = work[target]
                # Remove columns with no usable values and constant features.
                X = X.dropna(axis=1, how="all")
                X = X.loc[:, X.nunique(dropna=True) > 1]
                if X.shape[1] == 0:
                    st.error("There are no usable feature columns after removing empty or constant columns.")
                elif task == "Classification" and y.nunique() < 2:
                    st.error("Classification needs at least two target classes.")
                elif task == "Classification" and y.value_counts().min() < 2:
                    st.error("Each class needs at least two rows so the train/test split can work. Add more data or choose another target.")
                else:
                    try:
                        stratify = y if task == "Classification" else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42, stratify=stratify
                        )
                        candidates = model_candidates(task)
                        results = {}
                        progress = st.progress(0, text="Preparing the experiment…")
                        for i, (name, estimator) in enumerate(candidates.items()):
                            pipe = Pipeline([
                                ("prep", make_preprocessor(X_train)),
                                ("model", estimator),
                            ])
                            pipe.fit(X_train, y_train)
                            pred = pipe.predict(X_test)
                            if task == "Classification":
                                metrics = {
                                    "Accuracy": accuracy_score(y_test, pred),
                                    "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                                    "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                                    "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
                                }
                                score = metrics["F1 (weighted)"]
                            else:
                                metrics = {
                                    "R²": r2_score(y_test, pred),
                                    "MAE": mean_absolute_error(y_test, pred),
                                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                                }
                                score = metrics["R²"]
                            results[name] = {"pipeline": pipe, "metrics": metrics, "score": float(score)}
                            progress.progress((i+1)/len(candidates), text=f"Tested {name}")
                        best_name = max(results, key=lambda k: results[k]["score"])
                        st.session_state.models = results
                        st.session_state.best_name = best_name
                        st.session_state.best_pipeline = results[best_name]["pipeline"]
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.predictions = results[best_name]["pipeline"].predict(X_test)
                        st.session_state.target = target
                        st.session_state.task = task
                        st.session_state.run_at = datetime.now().strftime("%Y-%m-%d %H:%M")
                        progress.empty()
                        st.success(f"Experiment complete. Top baseline: {best_name}.")
                    except Exception as exc:
                        st.error(f"The experiment could not finish: {exc}")
            if st.session_state.models:
                st.markdown("### Experiment leaderboard")
                score_label = "F1 (weighted)" if st.session_state.task == "Classification" else "R²"
                rows = []
                for name, item in st.session_state.models.items():
                    row = {"Model": name, **item["metrics"]}
                    rows.append(row)
                board = pd.DataFrame(rows).sort_values(score_label, ascending=False)
                st.dataframe(board.style.format(precision=3), use_container_width=True, hide_index=True)
                st.success(f"Current leader: **{st.session_state.best_name}** · ranked by {score_label}.")
                st.caption("This is a baseline comparison on one train/test split—not a guarantee of future performance. For serious use, add cross-validation and tune the models.")

elif page == "Model Report":
    st.markdown('<div class="eyebrow">04 / READ THE RESULTS</div>', unsafe_allow_html=True)
    st.title("Model Report")
    if not st.session_state.models:
        st.info("Run an experiment in Model Forge to see model metrics and diagnostics.")
    else:
        best = st.session_state.best_name
        task = st.session_state.task
        item = st.session_state.models[best]
        st.markdown(f"### {best}")
        cols = st.columns(len(item["metrics"]))
        for col, (name, value) in zip(cols, item["metrics"].items()):
            with col:
                if name in ["MAE", "RMSE"]:
                    st.metric(name, f"{value:,.3f}")
                else:
                    st.metric(name, f"{value:.3f}")
        y_test = st.session_state.y_test
        pred = st.session_state.predictions
        if task == "Classification":
            labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
            # Use sklearn's display with actual labels; use plotly for consistent app styling.
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test.astype(str), pd.Series(pred).astype(str), labels=labels)
            fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Teal",
                            labels=dict(x="Predicted", y="Actual", color="Count"), title="Confusion matrix")
        else:
            plot_df = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(pred)})
            fig = px.scatter(plot_df, x="Actual", y="Predicted", title="Actual vs predicted", opacity=.75)
            lo = min(plot_df.min())
            hi = max(plot_df.max())
            fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Ideal fit", line=dict(dash="dash")))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Export your experiment")
        report = {
            "project": "FORGE AI",
            "dataset": st.session_state.filename,
            "target": st.session_state.target,
            "task": task,
            "best_baseline": best,
            "metrics": item["metrics"],
            "run_at": st.session_state.run_at,
            "note": "Baseline results from a single train/test split. Validate further before real-world use.",
        }
        a,b = st.columns(2)
        with a:
            model_buffer = io.BytesIO()
            joblib.dump(st.session_state.best_pipeline, model_buffer)
            st.download_button("↓ Download trained model (.joblib)",
                data=model_buffer.getvalue(),
                file_name="forge_ai_model.joblib", mime="application/octet-stream", use_container_width=True)
        with b:
            st.download_button("↓ Download experiment report (.json)",
                data=json.dumps(report, indent=2, default=str),
                file_name="forge_ai_report.json", mime="application/json", use_container_width=True)
        pred_df = pd.DataFrame({"Actual": np.asarray(y_test), "Predicted": np.asarray(pred)})
        st.download_button("↓ Download test predictions (.csv)", data=pred_df.to_csv(index=False),
            file_name="forge_ai_predictions.csv", mime="text/csv")
        st.caption("The exported model expects the same feature columns used during training. Keep a copy of the dataset schema with it.")
