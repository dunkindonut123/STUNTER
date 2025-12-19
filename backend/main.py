import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # IMPORT TAMBAHAN
import sys
import os

# ==========================================
# KONFIGURASI
# ==========================================
NGROK_TOKEN = "36YhkH596Yv9v6shYutYb2dii6e_2mDN1v84FpMZ8ttqDHqry"
CSV_FILE = 'final_stunting_wasting_gender_fixed.csv'

# Set Auth Token
ngrok.set_auth_token(NGROK_TOKEN)

# ==========================================
# 1. LOAD, SPLIT & TRAIN MODEL
# ==========================================
print("⏳ Memuat data, melatih AI, dan menghitung evaluasi...")

# Cek apakah file ada
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: File '{CSV_FILE}' tidak ditemukan!")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# Cleaning & Standardisasi
df = df.rename(columns={'Jenis Kelamin': 'sex', 'Umur (bulan)': 'age_months', 'Tinggi Badan (cm)': 'height_cm', 'Berat Badan (kg)': 'weight_kg'})
df['sex'] = df['sex'].astype(str).str.strip().str.lower().map({'laki-laki': 'M', 'perempuan': 'F', 'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'})
df['sex_encoded'] = df['sex'].apply(lambda x: 1 if x == 'M' else 0)

# Target Labels
stunt_keywords = ['Stunted', 'Severely Stunted']
waste_keywords = ['Underweight', 'Severely Underweight', 'Severely underweight']
df['target_stunting'] = df['Stunting'].apply(lambda x: 1 if x in stunt_keywords else 0)
df['target_wasting'] = df['Wasting'].apply(lambda x: 1 if any(k in str(x) for k in waste_keywords) else 0)

# Features & Targets
X = df[['age_months', 'height_cm', 'weight_kg', 'sex_encoded']]
y_stunt = df['target_stunting']
y_waste = df['target_wasting']

# --- PERUBAHAN DISINI: SPLIT DATA ---
# Membagi data menjadi 80% Training dan 20% Testing untuk evaluasi
X_train, X_test, y_stunt_train, y_stunt_test, y_waste_train, y_waste_test = train_test_split(
    X, y_stunt, y_waste, test_size=0.2, random_state=42
)

# --- MODEL STUNTING ---
print("\n⚙️  Melatih Model Stunting...")
model_stunting = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model_stunting.fit(X_train, y_stunt_train)



# Evaluasi Stunting
y_pred_stunt = model_stunting.predict(X_test)
print(f"📊 Evaluasi Model Stunting (Akurasi: {accuracy_score(y_stunt_test, y_pred_stunt):.2f})")
print("-" * 50)
print(classification_report(y_stunt_test, y_pred_stunt, target_names=['Normal', 'Stunted']))
print("Confusion Matrix:")
print(confusion_matrix(y_stunt_test, y_pred_stunt))
print("-" * 50)

# --- MODEL WASTING ---
print("\n⚙️  Melatih Model Wasting...")
model_wasting = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model_wasting.fit(X_train, y_waste_train)

# Evaluasi Wasting
y_pred_waste = model_wasting.predict(X_test)
print(f"📊 Evaluasi Model Wasting (Akurasi: {accuracy_score(y_waste_test, y_pred_waste):.2f})")
print("-" * 50)
print(classification_report(y_waste_test, y_pred_waste, target_names=['Normal', 'Wasted']))
print("Confusion Matrix:")
print(confusion_matrix(y_waste_test, y_pred_waste))
print("-" * 50)

print("\n🔍 Menjalankan K-Fold Cross-Validation (K=5)...")

# 1. Validasi untuk Model Stunting
cv_scores_stunting = cross_val_score(model_stunting, X, y_stunt, cv=5)
print(f"✅ Stunting CV Accuracy: {cv_scores_stunting.mean():.4f} (+/- {cv_scores_stunting.std() * 2:.4f})")

# 2. Validasi untuk Model Wasting
cv_scores_wasting = cross_val_score(model_wasting, X, y_waste, cv=5)
print(f"✅ Wasting CV Accuracy: {cv_scores_wasting.mean():.4f} (+/- {cv_scores_wasting.std() * 2:.4f})")

print("\n✅ Model AI Siap & Tervalidasi!")

# ==========================================
# 2. FLASK API
# ==========================================
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sex_input = data.get('sex', '').upper()
        age = float(data.get('age_months', 0))
        height = float(data.get('height_cm', 0))
        weight = float(data.get('weight_kg', 0))
        
        sex_encoded = 1 if sex_input == 'M' else 0
        input_data = pd.DataFrame([[age, height, weight, sex_encoded]], 
                                  columns=['age_months', 'height_cm', 'weight_kg', 'sex_encoded'])
        
        is_stunted = int(model_stunting.predict(input_data)[0])
        prob_stunt = float(model_stunting.predict_proba(input_data)[0][1])
        is_wasted = int(model_wasting.predict(input_data)[0])
        prob_waste = float(model_wasting.predict_proba(input_data)[0][1])
        
        if is_stunted == 1 and is_wasted == 1:
            status = 'both'
            confidence = (prob_stunt + prob_waste) / 2
        elif is_stunted == 1:
            status = 'stunted'
            confidence = prob_stunt
        elif is_wasted == 1:
            status = 'wasted'
            confidence = prob_waste
        else:
            status = 'healthy'
            confidence = ((1-prob_stunt) + (1-prob_waste)) / 2

        return jsonify({
            'prediction': status,
            'confidence': confidence,
            'details': {
                'stunted': bool(is_stunted),
                'stunted_probability': prob_stunt,
                'wasted': bool(is_wasted),
                'wasting_probability': prob_waste
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    ngrok.kill()
    try:
        public_url = ngrok.connect(5001).public_url
        print(f"\n🚀 SERVER BERJALAN!")
        print(f"👉 Public URL: {public_url}")
        print("\nTekan CTRL+C di terminal untuk berhenti.")
        app.run(port=5001)
    except Exception as e:
        print(f"Error starting ngrok: {e}")