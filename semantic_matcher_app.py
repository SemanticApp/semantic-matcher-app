
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Page config
st.set_page_config(page_title="Semantic Matcher", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Explanation generator
def generate_explanation(text_a, text_b):
    shared = set(re.findall(r"\w+", text_a.lower())) & set(re.findall(r"\w+", text_b.lower()))
    keywords = [w for w in shared if len(w) > 3]
    explanation = f"Shared terms: {', '.join(keywords[:5])}."
    return explanation if keywords else "Matched based on semantic similarity."

# File loader
def load_dataframe(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return None

# UI
st.title("🔎 Enhanced Semantic Matching App")
st.markdown("Match semantically related rows between multiple files using AI, with optional location matching boost and keyword explanations.")

uploaded_files = st.file_uploader("Upload CSV or XLSX files (2 or more)", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    dfs = [load_dataframe(f) for f in uploaded_files]
    file_names = [f.name for f in uploaded_files]

    st.sidebar.header("⚙️ Matching Settings")
    top_n = st.sidebar.slider("🔢 Top matches per row", 1, 10, 3)
    threshold = st.sidebar.slider("📊 Similarity threshold", 0.0, 1.0, 0.6, 0.01)

    selected_pairs = st.multiselect(
        "Select file pairs to compare (format: FileA | FileB)",
        [f"{file_names[i]} | {file_names[j]}" for i in range(len(file_names)) for j in range(len(file_names)) if i != j]
    )

    if selected_pairs:
        for pair in selected_pairs:
            file_a_name, file_b_name = pair.split(" | ")
            idx_a, idx_b = file_names.index(file_a_name), file_names.index(file_b_name)
            df_a, df_b = dfs[idx_a], dfs[idx_b]

            st.subheader(f"🆚 Comparing: {file_a_name} ↔ {file_b_name}")

            cols_a = st.multiselect(f"Select columns from {file_a_name}", df_a.columns.tolist(), key=f"a_cols_{file_a_name}")
            loc_col_a = st.selectbox(f"(Optional) Location column in {file_a_name}", ["None"] + df_a.columns.tolist(), key=f"a_loc_{file_a_name}")

            cols_b = st.multiselect(f"Select columns from {file_b_name}", df_b.columns.tolist(), key=f"b_cols_{file_b_name}")
            loc_col_b = st.selectbox(f"(Optional) Location column in {file_b_name}", ["None"] + df_b.columns.tolist(), key=f"b_loc_{file_b_name}")

            if cols_a and cols_b and st.button(f"Run Matching for {file_a_name} ↔ {file_b_name}"):
                with st.spinner("Encoding and matching rows..."):
                    text_a = df_a[cols_a].astype(str).agg(" ".join, axis=1).tolist()
                    text_b = df_b[cols_b].astype(str).agg(" ".join, axis=1).tolist()

                    emb_a = model.encode(text_a, convert_to_tensor=True)
                    emb_b = model.encode(text_b, convert_to_tensor=True)

                    similarity = cosine_similarity(emb_a.cpu(), emb_b.cpu())

                    results = []
                    progress = st.progress(0)
                    for i, row_sim in enumerate(similarity):
                        loc_a_val = str(df_a.loc[i, loc_col_a]) if loc_col_a != "None" else None
                        ranked_idx = np.argsort(-row_sim)
                        for j in ranked_idx[:top_n * 2]:  # check more to allow threshold filtering
                            score = row_sim[j]
                            if score >= threshold:
                                loc_b_val = str(df_b.loc[j, loc_col_b]) if loc_col_b != "None" else None
                                bonus = 0.05 if loc_a_val and loc_b_val and loc_a_val.lower() == loc_b_val.lower() else 0
                                explanation = generate_explanation(text_a[i], text_b[j])

                                results.append({
                                    "File A Index": i,
                                    "File B Index": j,
                                    "File A Row": text_a[i],
                                    "File B Row": text_b[j],
                                    "Similarity Score": min(score + bonus, 1.0),
                                    "Location Match Boost": bonus > 0,
                                    "Explanation": explanation
                                })
                        progress.progress((i + 1) / len(similarity))

                    result_df = pd.DataFrame(results).sort_values("Similarity Score", ascending=False)

                    # Remove duplicate File A Rows to keep only best match
                    result_df = result_df.drop_duplicates(subset=["File A Row"], keep="first")

                    st.success("✅ Matching complete.")
                    st.dataframe(result_df)

                    # Heatmap
                    st.subheader("🔍 Similarity Heatmap (Top 20 Rows)")
                    heatmap_data = similarity[:20, :20]
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(heatmap_data, annot=False, cmap="YlGnBu", ax=ax)
                    ax.set_title("Cosine Similarity Heatmap (Top 20x20)")
                    st.pyplot(fig)

                    # CSV download
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Matches", csv, f"{file_a_name}_vs_{file_b_name}_matches.csv", "text/csv")
else:
    st.info("Please upload at least two files to begin.")
