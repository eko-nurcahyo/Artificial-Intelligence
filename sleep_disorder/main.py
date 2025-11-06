from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Inisialisasi Flask
app = Flask(__name__)

# =====================================================
# 1️⃣ Load model, encoder, dan urutan kolom
# =====================================================
try:
    # Pastikan file-file ini ada di direktori yang sama dengan main.py
    model_lgbm = joblib.load("model_lightgbm.pkl")
    model_cat = joblib.load("model_catboost.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    # Memuat urutan kolom yang disimpan dari train_model.py
    feature_columns = joblib.load("feature_columns.pkl")
    print("✅ Model, encoder, dan daftar kolom berhasil dimuat.")

except FileNotFoundError as e:
    print("="*50)
    print(f"❌ ERROR: File model tidak ditemukan: {e}")
    print("➡️ Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu untuk menghasilkan file .pkl.")
    print("="*50)
    exit() # Keluar jika file penting tidak ada
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# =====================================================
# 2️⃣ Routing halaman utama
# =====================================================
@app.route('/')
def home():
    # Mengirimkan daftar kelas ke template (meskipun index.html ini tidak menggunakannya)
    classes = label_encoders['Sleep Disorder'].classes_
    # Flask akan mencari file ini di dalam folder 'templates'
    return render_template('index.html', disorder_classes=list(classes))

# =====================================================
# 3️⃣ Endpoint untuk prediksi (dengan probabilitas)
# =====================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Menerima data: {data}")

        # Ambil input dari form
        # Pastikan data yang diterima dari JS adalah angka
        usia = float(data['usia'])
        durasi = float(data['durasiTidur'])
        kualitas = float(data['kualitasTidur'])
        stres = float(data['stres'])

        # Nilai default untuk fitur lain (bisa dikembangkan nanti)
        # Ini adalah data mentah (string) sebelum di-encode
        sample_data = {
            'Gender': 'Male',  # Nilai default
            'Age': usia,
            'Occupation': 'Nurse', # Nilai default
            'Sleep Duration': durasi,
            'Quality of Sleep': kualitas,
            'Physical Activity Level': 30,  # Nilai default
            'Stress Level': stres,
            'BMI Category': 'Overweight', # Nilai default
            'Blood Pressure': '120/80', # Nilai default
            'Heart Rate': 80,  # Nilai default
            'Daily Steps': 6000 # Nilai default
        }

        # --- Proses data baru untuk prediksi ---

        # 1. Buat DataFrame dari data input
        input_df = pd.DataFrame([sample_data])

        # 2. Lakukan encoding pada data baru menggunakan encoder yang sudah disimpan
        encoded_df = input_df.copy()
        for col in label_encoders:
            if col in encoded_df.columns: # Hanya encode kolom yang ada di fitur
                val = encoded_df[col].iloc[0]
                try:
                    # Mengubah nilai string/numerik menjadi label (angka)
                    encoded_df[col] = label_encoders[col].transform([val])[0]
                except ValueError:
                    # Jika nilai tidak ada di encoder (misal: '130/90' atau 'Programmer')
                    print(f"Warning: Nilai '{val}' tidak ada di encoder untuk '{col}'. Menggunakan nilai 0 sebagai default.")
                    encoded_df[col] = 0 # Set ke nilai default (misal 0)
                except Exception as e:
                    print(f"Error encoding {col} with value {val}: {e}")
                    encoded_df[col] = 0 # Default jika ada error lain

        # 3. Ubah semua kolom menjadi tipe data yang benar (float)
        encoded_df = encoded_df.astype(float)

        # 4. Pastikan urutan kolom sama persis dengan saat training
        final_df_to_predict = encoded_df[feature_columns]
        # --- Selesai memproses data ---


        # =====================================================
        # Prediksi label dan probabilitas
        # =====================================================
        
        pred_lgbm = model_lgbm.predict(final_df_to_predict)[0]
        prob_lgbm = model_lgbm.predict_proba(final_df_to_predict)[0]

        # --- PERBAIKAN DI SINI ---
        # CatBoost .predict() mengembalikan array 2D, e.g., [[2]]
        # Kita perlu mengambil elemen pertama dari array pertama: [0][0]
        pred_cat = model_cat.predict(final_df_to_predict)[0][0]
        prob_cat = model_cat.predict_proba(final_df_to_predict)[0]

        # Decode label (mengubah 0, 1, 2 kembali ke "Insomnia", "Sleep Apnea", "Tidak Ada")
        classes = label_encoders['Sleep Disorder'].classes_
        
        # Konversi ke integer standar Python sebelum mengirim ke JSON
        hasil_lgbm = classes[int(pred_lgbm)]
        hasil_cat = classes[int(pred_cat)]

        # Buat dictionary hasil dengan persentase tiap kelas
        prob_dict_lgbm = {
            cls: round(prob_lgbm[i] * 100, 2) for i, cls in enumerate(classes)
        }
        prob_dict_cat = {
            cls: round(prob_cat[i] * 100, 2) for i, cls in enumerate(classes)
        }

        # =====================================================
        # Gabungkan hasil model
        # =====================================================
        if hasil_lgbm == hasil_cat:
            result_text = f"✅ Kedua model sepakat bahwa kemungkinan Anda mengalami <strong>{hasil_cat}</strong>."
        else:
            result_text = (
                f"⚠️ Model memberikan hasil berbeda.<br>"
                f"<b>LightGBM:</b> {hasil_lgbm} &nbsp;&nbsp; "
                f"<b>CatBoost:</b> {hasil_cat}."
            )

        # Return JSON lengkap
        return jsonify({
            'status': 'success',
            'hasil': result_text,
            'detail': {
                'LightGBM': {
                    'prediksi': hasil_lgbm,
                    'probabilitas': prob_dict_lgbm
                },
                'CatBoost': {
                    'prediksi': hasil_cat,
                    'probabilitas': prob_dict_cat
                }
            }
        })

    except Exception as e:
        print(f"Error di endpoint /predict: {e}")
        # PERBAIKAN: Menggunakan str(e) untuk mengonversi error menjadi string
        return jsonify({'status': 'error', 'message': f"Terjadi kesalahan server: {str(e)}"})

# =====================================================
# 4️⃣ Jalankan Flask App
# =====================================================
if __name__ == '__main__':
    # Pastikan untuk membuat folder 'templates' di direktori yang sama
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Folder 'templates' dibuat. Pastikan 'index.html' ada di dalamnya.")
    
    app.run(debug=True, port=5000)

