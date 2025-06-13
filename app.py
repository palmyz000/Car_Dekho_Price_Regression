import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re # สำหรับใช้ Regex เพื่อแยก Brand จาก Car_Name

# --- แก้ไขตรงนี้: ย้าย st.set_page_config() ขึ้นมาด้านบนสุด ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)
# -----------------------------------------------------------


# --- 1. Load Model and Feature Names (โหลดโมเดลและชื่อฟีเจอร์) ---
# ใช้ st.cache_resource เพื่อให้โหลดโมเดลแค่ครั้งเดียวเมื่อแอปเริ่มทำงาน
@st.cache_resource
def load_resources():
    try:
        pipeline = joblib.load('car_price_prediction_pipeline.joblib')
        feature_names = joblib.load('feature_names.joblib')
        return pipeline, feature_names
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'car_price_prediction_pipeline.joblib' and 'feature_names.joblib' are in the same directory as app.py.")
        st.stop() # หยุดการทำงานของแอปหากไฟล์ไม่พบ

model_pipeline, feature_names_list = load_resources()


# --- 2. Define Prediction Function (ฟังก์ชันทำนายราคา) ---
# (คัดลอกมาจาก Step 9, ปรับปรุงเล็กน้อยเพื่อให้เหมาะสมกับ Streamlit)
def predict_car_price(data: dict, model_pipeline, feature_names_list):
    """
    ฟังก์ชันสำหรับทำนายราคาขายรถยนต์จากข้อมูลดิบที่ผู้ใช้ป้อนเข้ามา

    Args:
        data (dict): Dictionary ที่มีข้อมูลรถยนต์จาก Streamlit UI
                     เช่น {'car_name': 'Honda City', 'year': 2017,
                           'km_driven': 50000, 'fuel': 'Petrol',
                           'seller_type': 'Individual',
                           'transmission': 'Manual', 'owner': 'First Owner'}
        model_pipeline: Pipeline ของโมเดลที่โหลดมา
        feature_names_list: รายชื่อคอลัมน์ features ที่โมเดลคาดหวัง

    Returns:
        float: ราคาทำนายเป็น INR
    """
    # 1. สร้าง DataFrame จากข้อมูล input
    df_input = pd.DataFrame([{
        'Car_Name': data['car_name'],
        'Year': data['year'],
        'selling_price': 0, # Placeholder, will be dropped
        'km_driven': data['km_driven'],
        'fuel': data['fuel'],
        'seller_type': data['seller_type'],
        'transmission': data['transmission'],
        'owner': data['owner']
    }])

    # 2. Preprocessing (ตามขั้นตอนที่เราทำมา)
    current_year = 2020 # ใช้ปีปัจจุบันที่ใช้ในการฝึกโมเดล
    df_input['Car_Age'] = current_year - df_input['Year']
    df_input.drop(['Year'], axis=1, inplace=True, errors='ignore')

    # สร้าง 'brand' feature จาก 'Car_Name'
    if 'Car_Name' in df_input.columns:
        df_input['brand'] = df_input['Car_Name'].apply(lambda x: re.match(r'^\w+', x).group(0) if re.match(r'^\w+', x) else 'Unknown')
        df_input.drop('Car_Name', axis=1, inplace=True)

    # จัดการ 'owner' (Ordinal Encoding)
    owner_mapping = {
        'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4
    }
    df_input['owner'] = df_input['owner'].map(owner_mapping)
    if df_input['owner'].isnull().any():
        st.warning("Warning: Invalid 'Owner' input received. Ensure it matches 'First Owner', 'Second Owner', etc.")

    # One-Hot Encoding สำหรับ Categorical Features
    categorical_cols_onehot = ['brand', 'fuel', 'seller_type', 'transmission']
    df_input_encoded = pd.get_dummies(df_input, columns=categorical_cols_onehot, drop_first=True, dtype=int)

    # จัดการคอลัมน์ที่อาจจะหายไปหรือเกินมาจากการ One-Hot Encoding
    df_final_features = pd.DataFrame(columns=feature_names_list)
    for col in df_final_features.columns:
        if col in df_input_encoded.columns:
            df_final_features[col] = df_input_encoded[col]
        else:
            df_final_features[col] = 0

    if 'selling_price' in df_final_features.columns:
        df_final_features.drop('selling_price', axis=1, inplace=True, errors='ignore')

    df_final_features = df_final_features[feature_names_list]

    # 3. ทำนายด้วย Pipeline
    predicted_log_price = model_pipeline.predict(df_final_features)[0]

    # 4. แปลงค่าทำนายกลับสู่สเกลราคาจริง (INR)
    predicted_original_price = np.expm1(predicted_log_price)

    return predicted_original_price


# --- 3. Streamlit UI (โค้ดส่วนนี้ยังคงเหมือนเดิม) ---
st.title("🚗 Car Price Predictor")
st.markdown("Enter the car details below to get an estimated selling price.")

# Input fields
car_name = st.text_input("Car Name (e.g., Maruti Swift Dzire VDI)", "Honda City")
year = st.slider("Year of Manufacture", min_value=1990, max_value=2020, value=2017)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
fuel = st.selectbox("Fuel Type", ('Petrol', 'Diesel', 'CNG'))
seller_type = st.selectbox("Seller Type", ('Individual', 'Dealer', 'Trustmark Dealer'))
transmission = st.selectbox("Transmission Type", ('Manual', 'Automatic'))
owner = st.selectbox("Owner Type", ('First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'))


# Prediction button
if st.button("Predict Selling Price"):
    input_data = {
        'car_name': car_name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }

    try:
        predicted_price = predict_car_price(input_data, model_pipeline, feature_names_list)
        st.success(f"**Predicted Selling Price: {predicted_price:,.2f} INR**")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check your input values.")


st.markdown("---")
st.markdown("This predictor is based on a machine learning model trained on a dataset of used cars.")
st.markdown("Created by Your Name/Team Name")