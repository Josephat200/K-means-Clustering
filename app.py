import streamlit as st

st.set_page_config(page_title="Iris K-Means", page_icon="🌸")

st.title("🌸 Iris K-Means Clustering App")
st.markdown(
    """
    Welcome! This app demonstrates **unsupervised clustering** of the Iris dataset
    using the **K-Means algorithm**.

    Use the main app (`machine.py`) to:
    - Predict which cluster a new flower belongs to
    - Explore cluster visualizations
    - View model metrics and statistics
    """
)
