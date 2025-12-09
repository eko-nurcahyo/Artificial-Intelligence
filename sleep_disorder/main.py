from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# -------------------------
# Load artifacts
# -------------------------
try:
    model_lgbm = joblib.load("model_lightgbm.pkl")
    model_cat = joblib.load("model_catboost.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")  # order saat training
    print("✅ Model, encoder, dan daftar kolom berhasil dimuat.")
except FileNotFoundError as e:
    print("=" * 60)
    print(f"❌ ERROR: File model tidak ditemukan: {e}")
    print("➡️ Jalankan train_model.py terlebih dahulu.")
    print("=" * 60)
    raise
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    raise

# -------------------------
# Default values for non-input features (keputusan Anda)
# -------------------------
DEFAULTS = {
    'Gender': 'Male',
    'Occupation': 'Nurse',
    'Physical Activity Level': 30,
    'BMI Category': 'Overweight',
    'Blood Pressure': '120/80',
    'Heart Rate': 80,
    'Daily Steps': 6000
}

# Utility: safe encode a single column value using LabelEncoder-like object
def safe_encode(col_name, val):
    """
    Try to transform val using label_encoders[col_name].
    If val not found in encoder.classes_, fall back to encoder.classes_[0].
    If encoder not present, return val unchanged.
    """
    if col_name not in label_encoders:
        return val
    le = label_encoders[col_name]
    # If incoming is numeric and encoder classes are numeric-like strings, try convert
    try:
        # check if val present
        classes = list(le.classes_)
        if val in classes:
            return int(le.transform([val])[0])
        # try string cast (some encoders have '120/80' etc.)
        if str(val) in classes:
            return int(le.transform([str(val)])[0])
        # fallback: use first class available (safe and valid)
        fallback = classes[0]
        return int(le.transform([fallback])[0])
    except Exception:
        # As last resort, return 0 (but this should rarely happen)
        return 0

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    classes = label_encoders['Sleep Disorder'].classes_
    return render_template('index.html', disorder_classes=list(classes))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # print for server logs (helps debugging)
        print("Received data:", data)

        # -------------------------
        # REQUIRED inputs (from user)
        # -------------------------
        # Validate and convert
        try:
            usia = float(data.get('usia', None))
            durasi = float(data.get('durasiTidur', None))
            kualitas = float(data.get('kualitasTidur', None))
            stres = float(data.get('stres', None))
        except Exception as e:
            return jsonify({'status': 'error', 'message': 'Input tidak valid. Pastikan usia, durasiTidur, kualitasTidur, dan stres berisi angka.'}), 400

        # -------------------------
        # Build sample using 4 inputs + defaults
        # -------------------------
        sample = {
            'Gender': DEFAULTS['Gender'],
            'Age': usia,
            'Occupation': DEFAULTS['Occupation'],
            'Sleep Duration': durasi,
            'Quality of Sleep': kualitas,
            'Physical Activity Level': DEFAULTS['Physical Activity Level'],
            'Stress Level': stres,
            'BMI Category': DEFAULTS['BMI Category'],
            'Blood Pressure': DEFAULTS['Blood Pressure'],
            'Heart Rate': DEFAULTS['Heart Rate'],
            'Daily Steps': DEFAULTS['Daily Steps']
        }

        # -------------------------
        # Prepare DataFrame with feature_columns order
        # -------------------------
        input_df = pd.DataFrame([sample], columns=feature_columns)

        # -------------------------
        # Encode categorical columns present in label_encoders (skip target)
        # -------------------------
        for col in input_df.columns:
            if col in label_encoders and col != 'Sleep Disorder':
                encoded_val = safe_encode(col, input_df.loc[0, col])
                input_df.loc[0, col] = encoded_val

        # Convert all columns to numeric type expected by models
        # If some columns are still non-numeric, attempt conversion
        for c in input_df.columns:
            try:
                input_df[c] = pd.to_numeric(input_df[c])
            except Exception:
                # fallback: fill with 0
                input_df[c] = 0

        # Reorder columns to exact training order and ensure no missing cols
        final_df_to_predict = input_df[feature_columns].astype(float)

        # -------------------------
        # Predict LightGBM (robust)
        # -------------------------
        pred_lgbm_raw = model_lgbm.predict(final_df_to_predict)
        # ensure scalar integer
        pred_lgbm = int(np.array(pred_lgbm_raw).reshape(-1)[0])

        prob_lgbm_raw = model_lgbm.predict_proba(final_df_to_predict)
        prob_lgbm = np.array(prob_lgbm_raw).reshape(len(prob_lgbm_raw), -1)[0]

        # -------------------------
        # Predict CatBoost (robust to output shape differences)
        # -------------------------
        # Prediction
        pred_cat_raw = model_cat.predict(final_df_to_predict)
        # catboost may return [[2]] or [2] depending on version; normalize:
        pred_cat_arr = np.array(pred_cat_raw).reshape(-1)
        pred_cat = int(pred_cat_arr[0])

        # Probabilities
        try:
            prob_cat_raw = model_cat.predict_proba(final_df_to_predict)
            prob_cat = np.array(prob_cat_raw).reshape(len(prob_cat_raw), -1)[0]
        except Exception:
            # If predict_proba not available or shape odd, set uniform as fallback
            n_classes = len(label_encoders['Sleep Disorder'].classes_)
            prob_cat = np.ones(n_classes) / n_classes

        # -------------------------
        # Decode labels
        # -------------------------
        classes = list(label_encoders['Sleep Disorder'].classes_)
        hasil_lgbm = classes[pred_lgbm] if 0 <= pred_lgbm < len(classes) else "Unknown"
        hasil_cat = classes[pred_cat] if 0 <= pred_cat < len(classes) else "Unknown"

        # Build probability dicts with percentages
        prob_dict_lgbm = {cls: round(float(prob_lgbm[i]) * 100, 2) for i, cls in enumerate(classes)}
        prob_dict_cat = {cls: round(float(prob_cat[i]) * 100, 2) for i, cls in enumerate(classes)}

        # Compose result text
        if hasil_lgbm == hasil_cat:
            result_text = f"✅ Kedua model sepakat bahwa kemungkinan Anda mengalami <strong>{hasil_cat}</strong>."
        else:
            result_text = (
                f"⚠️ Model memberikan hasil berbeda.<br>"
                f"<b>LightGBM:</b> {hasil_lgbm} &nbsp;&nbsp; "
                f"<b>CatBoost:</b> {hasil_cat}."
            )

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
        print("Error di endpoint /predict:", e)
        return jsonify({'status': 'error', 'message': f"Terjadi kesalahan server: {str(e)}"}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Folder 'templates' dibuat. Pastikan 'index.html' ada di dalamnya.")
    app.run(debug=True, port=5000)
