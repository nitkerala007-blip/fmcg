"""
Streamlit app for interactive exploration of the FMCG Hybrid Recommendation System.
Enhanced with product category filtering, animations, card hover effects, and publication-ready UI.
"""
import streamlit as st
st.set_page_config(page_title="FMCG Hybrid Recommender", layout="wide")

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns
import time

sns.set(style="whitegrid")

# ---------- Helpers and caching ----------
@st.cache_data(show_spinner=False)
def load_products(csv_path='BigBasket Products 2.csv'):
    if not os.path.exists(csv_path):
        st.error(f"Product CSV not found at: {csv_path}. Please upload or place the file in the app folder.")
        return None
    df = pd.read_csv(csv_path)
    required = ['product', 'category', 'sub_category', 'brand', 'sale_price', 'market_price', 'rating', 'description']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None
    df = df.dropna(subset=required).drop_duplicates(subset=['product']).reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=False)
def load_or_build_artifacts(df,
                            preproc_path='fmcb_preprocessors.joblib',
                            embeddings_path='product_embeddings.npy',
                            embed_dim=200,
                            tfidf_max=500):
    if os.path.exists(preproc_path) and os.path.exists(embeddings_path):
        saved = joblib.load(preproc_path)
        ohe, scaler, tfidf, svd = saved['ohe'], saved['scaler'], saved['tfidf'], saved['pca']
        product_embeddings = np.load(embeddings_path)
        product_embeddings = normalize(product_embeddings)
        return ohe, scaler, tfidf, svd, product_embeddings, None

    categorical_cols = [c for c in ['category', 'sub_category', 'brand', 'type'] if c in df.columns]
    num_cols = [c for c in ['sale_price', 'market_price', 'rating'] if c in df.columns]

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_matrix = ohe.fit_transform(df[categorical_cols])

    scaler = MinMaxScaler()
    num_matrix = scaler.fit_transform(df[num_cols].astype(float))

    tfidf = TfidfVectorizer(max_features=tfidf_max, stop_words='english')
    desc_tfidf = tfidf.fit_transform(df['description'].astype(str)).toarray()

    product_features = np.hstack([cat_matrix, num_matrix, desc_tfidf])

    svd = TruncatedSVD(n_components=embed_dim, random_state=42)
    product_embeddings = svd.fit_transform(product_features)
    product_embeddings = normalize(product_embeddings)

    joblib.dump({'ohe': ohe, 'scaler': scaler, 'tfidf': tfidf, 'pca': svd}, preproc_path)
    np.save(embeddings_path, product_embeddings)
    return ohe, scaler, tfidf, svd, product_embeddings, product_features

@st.cache_resource(show_spinner=False)
def build_content_nn(product_embeddings, top_k=100):
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, product_embeddings.shape[0]), metric='cosine', n_jobs=-1)
    nn.fit(product_embeddings)
    distances, indices = nn.kneighbors(product_embeddings, return_distance=True)
    neighbor_indices = indices[:, 1:top_k + 1]
    neighbor_distances = distances[:, 1:top_k + 1]
    return nn, neighbor_indices, neighbor_distances

@st.cache_data(show_spinner=False)
def build_popularity(df):
    pop = (df['rating'].astype(float).fillna(df['rating'].astype(float).mean()).values + 1.0)
    pop = pop / pop.sum()
    return pop

# ---------- Fixed Collaborative Filtering ----------
@st.cache_resource(show_spinner=False)
def build_synthetic_cf(_train_matrix, k=100):
    """Compute SVD-based user/item factors. Streamlit will skip hashing _train_matrix."""
    if _train_matrix is None or _train_matrix.shape[0] == 0:
        return None, None
    try:
        u, s, vt = svds(_train_matrix, k=min(k, _train_matrix.shape[1] - 1))
        s = s[::-1]
        u = u[:, ::-1]
        vt = vt[::-1, :]
        user_factors = u.dot(np.diag(s))
        item_factors = vt.T * s
        return user_factors, item_factors
    except Exception as e:
        st.warning(f"CF decomposition failed — fallback to popularity. Error: {e}")
        return None, None

# ---------- Load data ----------
st.title("🌟 FMCG Hybrid Recommendation System")
st.markdown("Explore products with content + collaborative recommendations, enhanced UI and animations.")

df = load_products()
if df is None:
    st.stop()

with st.spinner("Loading/Preparing artifacts..."):
    ohe, scaler, tfidf, svd, product_embeddings, product_features = load_or_build_artifacts(df)

nn, neighbor_indices, neighbor_distances = build_content_nn(product_embeddings, top_k=200)
popularity = build_popularity(df)

# ---------- Load train_matrix.npz ----------
train_matrix_path = 'train_matrix.npz'
user_factors, item_factors = None, None

if os.path.exists(train_matrix_path):
    try:
        train_csr = sp.load_npz(train_matrix_path)
        user_factors, item_factors = build_synthetic_cf(train_csr, k=100)
        st.success("Collaborative filtering matrix loaded successfully.")
    except Exception as e:
        st.warning(f"Failed to load train_matrix.npz — CF fallback to popularity. Error: {e}")
else:
    st.info("No train_matrix.npz file found. Using popularity-based fallback.")

# ---------- UI ----------
st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filter by Category")
    categories = df['category'].unique()
    selected_category = st.selectbox("Select category", categories)

    filtered_products = df[df['category'] == selected_category]
    st.subheader("Choose Product")
    product_list = filtered_products['product'].tolist()
    selected_product = st.selectbox("Search product", product_list)

    st.markdown("**Recommendation Parameters**")
    top_k = st.slider("Number of recommendations", 3, 30, 10)
    alpha = st.slider("Hybrid weight (α): 0=Content, 1=CF", 0.0, 1.0, 0.5, 0.05)
    run_button = st.button("✨ Get Recommendations", key="run_button")

with col2:
    st.subheader("Product Details")
    sel_idx = df.index[df['product'] == selected_product].tolist()
    if len(sel_idx) > 0:
        sel_idx = sel_idx[0]
        st.markdown(f"**Product:** {df.loc[sel_idx, 'product']}")
        st.markdown(f"Brand: {df.loc[sel_idx, 'brand']}")
        st.markdown(f"Category: {df.loc[sel_idx, 'category']} / {df.loc[sel_idx, 'sub_category']}")
        st.markdown(f"Price: {df.loc[sel_idx, 'sale_price']} | Rating: {df.loc[sel_idx, 'rating']}")
        st.write(df.loc[sel_idx, 'description'])
    else:
        st.warning("Selected product not found.")

# ---------- Recommendation Logic ----------
def get_content_scores_for_item(seed_idx, candidate_ids=None):
    if candidate_ids is None:
        candidates = np.arange(product_embeddings.shape[0])
    else:
        candidates = np.array(candidate_ids)
    seed_emb = product_embeddings[seed_idx].reshape(1, -1)
    cand_emb = product_embeddings[candidates]
    sim = cand_emb.dot(seed_emb.T).flatten()
    return sim

def get_cf_scores_for_user_proxy(seed_idx):
    if item_factors is not None:
        item_vec = item_factors[seed_idx]
        scores = item_factors.dot(item_vec)
        return scores
    else:
        return popularity

def hybrid_scores_for_item(seed_idx, alpha=0.5, candidate_ids=None):
    if candidate_ids is None:
        candidates = np.arange(product_embeddings.shape[0])
    else:
        candidates = np.array(candidate_ids)
    content_scores = get_content_scores_for_item(seed_idx, candidate_ids=candidates)
    cf_scores = get_cf_scores_for_user_proxy(seed_idx)[candidates]
    if content_scores.std() > 0:
        content_scores = (content_scores - content_scores.mean()) / content_scores.std()
    if cf_scores.std() > 0:
        cf_scores = (cf_scores - cf_scores.mean()) / cf_scores.std()
    hybrid = alpha * cf_scores + (1 - alpha) * content_scores
    return candidates, hybrid

# ---------- Display Recommendations ----------
if run_button:
    st.session_state["clicked"] = True
    with st.spinner("Computing recommendations..."):
        candidates, scores = hybrid_scores_for_item(sel_idx, alpha=alpha)
        scores[sel_idx] = -np.inf
        top_idx = np.argpartition(-scores, range(top_k))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        rec_df = df.loc[top_idx, ['product', 'brand', 'category', 'sale_price', 'rating']].copy()
        rec_df['score'] = scores[top_idx]

        st.subheader("Recommended Products")
        for i, row in rec_df.iterrows():
            st.markdown(f"""
            <div style='
                border:1px solid #ddd;
                padding:12px;
                margin-bottom:8px;
                border-radius:12px;
                transition: transform 0.2s, box-shadow 0.2s;
            ' onmouseover="this.style.transform='scale(1.03)'; this.style.boxShadow='0 8px 20px rgba(0,0,0,0.2)';"
              onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 0 0 rgba(0,0,0,0)';">
            <b>{row['product']}</b><br>
            Brand: {row['brand']} | Category: {row['category']}<br>
            Price: {row['sale_price']} | Rating: {row['rating']}<br>
            Score: {row['score']:.2f}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)

        # Content Neighbors
        st.subheader("Top Content Neighbors")
        content_nei = neighbor_indices[sel_idx][:5]
        cn_df = df.loc[content_nei, ['product', 'brand', 'category', 'sale_price', 'rating']].copy()
        st.dataframe(cn_df.reset_index(drop=True))

        # Scores chart
        st.subheader("Recommendation Scores")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(rec_df['product'].str.slice(0, 60), rec_df['score'], color='#4CAF50', alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Hybrid Score")
        st.pyplot(fig)

# ---------- Sidebar ----------
st.sidebar.title("Advanced / Deploy")
st.sidebar.markdown("""
- Upload your own `BigBasket Products 2.csv`.
- Upload `fmcb_preprocessors.joblib` + `product_embeddings.npy` for faster start.
""")

uploaded_csv = st.sidebar.file_uploader("Upload product CSV", type=['csv'])
if uploaded_csv:
    try:
        newdf = pd.read_csv(uploaded_csv)
        st.sidebar.success("Uploaded CSV. Press button to reload app with this dataset.")
        if st.sidebar.button("Reload with uploaded CSV"):
            newdf.to_csv('BigBasket Products 2.csv', index=False)
            st.experimental_rerun()
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))

st.sidebar.markdown("To deploy: push this folder to GitHub and connect to Streamlit Cloud.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<center>© 2025 Amal A. | All Rights Reserved</center>", unsafe_allow_html=True)