import streamlit as st
st.set_page_config(page_title="Semantic Matcher", layout="wide")

import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.title("📄 Semantic Matching App")
st.markdown("Match semantically related rows between any two files using AI.")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload File A (CSV/XLSX)", type=["csv", "xlsx"], key="file_a")
with col2:
    file_b = st.file_uploader("Upload File B (CSV/XLSX)", type=["csv", "xlsx"], key="file_b")

def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        return None

df_a = load_dataframe(file_a)
df_b = load_dataframe(file_b)

if df_a is not None and df_b is not None:
    st.sidebar.header("⚙️ Matching Settings")

    st.subheader("Step 2: Select Columns for Semantic Comparison")

    with st.expander("File A Columns"):
        cols_a = st.multiselect("Columns to use from File A", df_a.columns.tolist(), default=df_a.columns[:2].tolist(), key="cols_a")
        loc_col_a = st.selectbox("(Optional) Location column in File A", ["None"] + df_a.columns.tolist(), key="loc_a")

    with st.expander("File B Columns"):
        cols_b = st.multiselect("Columns to use from File B", df_b.columns.tolist(), default=df_b.columns[:2].tolist(), key="cols_b")
        loc_col_b = st.selectbox("(Optional) Location column in File B", ["None"] + df_b.columns.tolist(), key="loc_b")

    top_n = st.sidebar.slider("🔢 Number of top matches to show", 1, 10, 1)
    threshold = st.sidebar.slider("📊 Similarity threshold", 0.0, 1.0, 0.6, step=0.01)

    if cols_a and cols_b:
        if st.button("🔍 Run Semantic Matching"):
            with st.spinner("Matching in progress..."):
                text_a = df_a[cols_a].astype(str).agg(" ".join, axis=1).tolist()
                text_b = df_b[cols_b].astype(str).agg(" ".join, axis=1).tolist()

                emb_a = model.encode(text_a, convert_to_tensor=True)
                emb_b = model.encode(text_b, convert_to_tensor=True)

                similarity = cosine_similarity(emb_a.cpu(), emb_b.cpu())

                results = []
                for i, row_sim in enumerate(similarity):
                    loc_a_val = str(df_a.loc[i, loc_col_a]) if loc_col_a != "None" else None

                    ranked_idx = np.argsort(-row_sim)
                    matches = []
                    for idx in ranked_idx[:top_n * 2]:
                        score = row_sim[idx]
                        if score < threshold:
                            continue

                        loc_b_val = str(df_b.loc[idx, loc_col_b]) if loc_col_b != "None" else None
                        bonus = 0.05 if loc_a_val and loc_b_val and loc_a_val.lower() == loc_b_val.lower() else 0

                        matches.append({
                            "File A Row": text_a[i],
                            "Matched File B Row": text_b[idx],
                            "Similarity Score": min(score + bonus, 1.0),
                            "Location Match Boost": bonus > 0
                        })

                    matches = sorted(matches, key=lambda x: -x["Similarity Score"])[:top_n]
                    results.extend(matches)

                result_df = pd.DataFrame(results)
                st.success("✅ Matching complete.")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "semantic_matches.csv", "text/csv")
else:
    st.info("Please upload both File A and File B to begin.")