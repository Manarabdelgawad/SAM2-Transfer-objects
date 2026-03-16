import streamlit as st
import requests
from PIL import Image
import io
import time

st.set_page_config(page_title="SAM2 Transfer", layout="wide")
st.title(" Extract Object & Change Background")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Source Image")
    source_file = st.file_uploader("Upload object image", type=["jpg","jpeg","png"], key="s")

with col2:
    st.subheader("🌄 Background")
    bg_file = st.file_uploader("Upload background", type=["jpg","jpeg","png"], key="b")



if source_file and bg_file:
    if st.button("🚀 Transfer Object to Background", type="primary", use_container_width=True):
        with st.spinner("Processing... This can take 15-60 seconds (SAM2 is heavy)"):
            files = {
                "source": (source_file.name, source_file.getvalue(), source_file.type),
                "background": (bg_file.name, bg_file.getvalue(), bg_file.type)
            }
            try:
                response = requests.post("http://127.0.0.1:8000/transfer", files=files, timeout=300)
                
                if response.status_code == 200:
                    result = Image.open(io.BytesIO(response.content))
                    st.success("Success!")
                    st.image(result, use_column_width=True)
                    
                    buf = io.BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button("⬇️ Download Result", buf.getvalue(), "sam2_result.png", "image/png")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection failed. Make sure FastAPI is running.\nError: {e}")

