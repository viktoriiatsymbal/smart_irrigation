"""Smart Irrigation: one app, two views. Run from the repo root: streamlit run app/app.py"""
import streamlit as st

st.set_page_config(page_title="Smart Irrigation", layout="wide")

view = st.sidebar.radio("View", ["Farmer", "Developer"], horizontal=True, key="view")
st.sidebar.divider()

if view.endswith("Farmer"):
    import farmer_view
    _, mid, _ = st.columns([1, 2.4, 1])         # keep the farmer page narrow and readable
    with mid:
        farmer_view.render()
else:
    for k in ("f_field", "f_day", "f_system", "f_level"):      # keep the farmer's choices while viewing Developer
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    import dev_view
    dev_view.render()
