import streamlit as st
from prediction import predict
st.title("Fruit Freshness Detection")

uploaded_file = st.file_uploader('Upload the pic of Fruit',type=["jpg","jpeg","png"],accept_multiple_files=True)

for uploaded_image in uploaded_file:
    if uploaded_image is not None:
        st.image(uploaded_image)
        prediction = predict(uploaded_image)
        st.info(f"Fruit Freshness: {prediction}")