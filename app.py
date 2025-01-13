import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge

# Load Model SVM
@st.cache_resource
def load_model():
    with open("model/svm_sentiment_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("model/tfidf_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

# Hasil load model diubah ke dalam bentuk Vector
model, vectorizer = load_model()

# Inisiasi session
if "results" not in st.session_state:
    st.session_state["results"] = {"Positif": 0, "Negatif": 0}

# Streamlit app UI
st.title("✨ Sentiment Analysis with SVM ✨")

add_vertical_space(2)

st.write(
    "Masukkan teks di bawah untuk menganalisis sentimen dengan cepat dan akurat:",
    unsafe_allow_html=True
)
user_input = st.text_area("✍️ Teks Anda:", placeholder="Ketik teks di sini...", height=100)

if st.button("🔍 Analisis Sentimen"):
    if user_input.strip():
        # Merubah user input menjadi vector
        input_data = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_data)
        
        # Mapping sentimen label
        sentiment_label = {0: "Negatif", 1: "Positif"}  # Sesuaikan dengan model Anda
        sentiment = sentiment_label.get(prediction[0], "Tidak diketahui")
        
        # Update session 
        if sentiment in st.session_state["results"]:
            st.session_state["results"][sentiment] += 1
        
        st.success(f"🎉 Hasil Sentimen: **{sentiment}**")
    else:
        st.warning("⚠️ Silakan masukkan teks untuk analisis.")

# Tampilan Pie Chart
st.markdown("---")
st.subheader("📊 Statistik Sentimen")
if any(st.session_state["results"].values()):
    labels = list(st.session_state["results"].keys())
    sizes = list(st.session_state["results"].values())

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 12}
    )
    ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.

    st.pyplot(fig)
else:
    st.info("📭 Belum ada data untuk ditampilkan.")

# Footer 
st.markdown("---")
with st.container():
    st.caption("Aplikasi ini menggunakan model SVM untuk analisis sentimen.")
    st.caption("Dikembangkan dengan ❤️ oleh Saya.")
