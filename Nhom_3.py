import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="CNN Model", page_icon="🤖", layout="wide")

# Header gồm logo và tiêu đề
col1, col2, col3 = st.columns([1, 8, 1])

with col1:
    st.image("logo_ute.png", width=140, use_container_width=False)

with col2:
    st.markdown("""
        <div style="text-align:center; line-height:1.5;">
            <h2 style="color:#c62828; font-weight:bold; margin-bottom:5px;">
                TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP. HỒ CHÍ MINH
            </h2>
            <h3 style="color:#1565c0; font-weight:bold; margin-top:0;">
                KHOA CƠ KHÍ
            </h3>
            <h1 style="color:#00796b; font-weight:bold; text-shadow:1px 1px 2px #ccc; margin-top:15px;">
                NHẬN DIỆN GIÀY - DÉP - GIÀY CAO GÓT - ỦNG
            </h1>
            <p style="font-size:18px; color:#555; margin-top:8px;">
                Tải ảnh lên để mô hình nhận diện loại giày, dép, giày cao gót hoặc ủng nhé!
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("logo_khoa.jpg", width=140, use_container_width=False)

# CSS TOÀN TRANG
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fafafa, #e0f7fa);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Segoe UI', sans-serif;
    }
    .css-18e3th9 {
        padding-top: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# DANH SÁCH NHÃN
shoes_names = ['Dep', 'Giay', 'Giaycaogot', 'Ung']


# TẢI MODEL
model_path = 'Nhom_3.keras'  #################
if not os.path.exists(model_path):
    st.error("❌ Không tìm thấy model. Hãy chắc chắn file '2.keras' nằm cùng thư mục với file này.") #############
    st.stop()

model = load_model(model_path)


# HÀM DỰ ĐOÁN
def classify_images(image):
    img = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_expanded = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_expanded)
    result = tf.nn.softmax(predictions[0])
    class_index = np.argmax(result)
    confidence = float(np.max(result))

    # Nếu độ tin cậy thấp hơn 80% thì cho ra Unknown
    if confidence < 0.8:
        return "Unknown", confidence
    return shoes_names[class_index], confidence

# GIAO DIỆN 2 CỘT
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Chọn ảnh từ thiết bị:")
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)

with col2:
    if uploaded_file:
        with st.spinner("🔍 Đang nhận diện..."):
            label, confidence = classify_images(image)

        if label == "Unknown":
            st.error("⚠️ Không xác định được loại giày, dép, giày cao gót hoặc ủng (độ tin cậy thấp).")
        else:
            st.success(f"✅ Kết quả nhận diện: **{label}**")

        st.markdown(f"### 🎯 Độ tin cậy: `{confidence*100:.2f}%`")
        st.progress(confidence)


# FOOTER THÔNG TIN NHÓM
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:16px;'>
    <strong>Thực hiện bởi:</strong><br>
    Huynh Vo Phuc Loc <br>
    Nguyen Thien Nhan <br>
    <strong>Giảng viên hướng dẫn:</strong><br>
    TS. Cai Viet Anh Dung<br>
    <em>Đại học Công nghệ Kỹ thuật TP. HCM</em>
</div>
""", unsafe_allow_html=True)