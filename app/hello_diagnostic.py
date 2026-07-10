"""Minimal diagnostic script with zero project dependencies.

Used to isolate whether a Streamlit Cloud segmentation fault is caused by
this project's dependencies or by the hosting environment itself. Deploy
this file (not app/dashboard.py) as a separate throwaway app to test.
"""

import streamlit as st

st.write("Hello, world! If you can see this, Streamlit itself is working.")
