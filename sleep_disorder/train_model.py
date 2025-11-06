import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import matplotlib
matplotlib.use("Agg") # Mencegah error plotting di environment tanpa GUI

print("--- TAHAP 1: MEMUAT DATA ---")
# 1.1 Ambil Dataset dari Kaggle
print("🔄 Mengambil dataset dari Kaggle...")
file_path = "Sleep_health_and_lifestyle_dataset.csv"

# Menggunakan .load_dataset (seperti kode asli Anda)
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mdsultanulislamovi/sleep-disorder-diagnosis-dataset",
    file_path
)

print("✅ Dataset berhasil dimuat dari Kaggle!")
print(df.head(), "\n")

# =====================================================
# 2️⃣ Cek Missing Value & Bersihkan Data
# =====================================================
print("--- TAHAP 2: MEMBERSIHKAN DATA ---")
print("🔍 Mengecek missing value (sebelum diperbaiki)...")
print(df.isnull().sum(), "\n")

# --- PERBAIKAN PENTING ADA DI SINI ---
# Mengganti nilai NaN (kosong) di kolom 'Sleep Disorder' dengan 'Tidak Ada'
# Ini penting agar data 'Tidak Ada' (pasien sehat) ikut dilatih
print("Mengganti NaN di 'Sleep Disorder' menjadi 'Tidak Ada'...")
df['Sleep Disorder'].fillna('Tidak Ada', inplace=True)

print(f"Distribusi 'Sleep Disorder' SETELAH diperbaiki:")
print(df['Sleep Disorder'].value_counts(dropna=False), "\n")

# Hapus baris jika ada NaN di kolom FITUR (jika ada)
df.dropna(inplace=True)
print(f"✅ Data cleaning selesai. Total data: {len(df)}\n")

# =====================================================
# 3️⃣ Label Encoding untuk Kolom Kategorikal
# =====================================================
print("--- TAHAP 3: ENCODING DATA ---")
label_encoders = {}
# Mengubah semua kolom non-numerik (object) menjadi angka
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    if col != 'Sleep Disorder': # Encode fitur
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"Kolom Fitur '{col}' di-encode.")

# Encode kolom target secara terpisah
le_target = LabelEncoder()
df['Sleep Disorder'] = le_target.fit_transform(df['Sleep Disorder'])
label_encoders['Sleep Disorder'] = le_target
print(f"Kolom Target 'Sleep Disorder' di-encode.")
print("Kelas target:", list(label_encoders['Sleep Disorder'].classes_))


print("✅ Encoding selesai.")
print(df.head(), "\n")

# =====================================================
# 4️⃣ Pisahkan Fitur & Target
# =====================================================
print("--- TAHAP 4: MEMBAGI DATASET ---")
# Hilangkan kolom yang tidak relevan untuk prediksi
X = df.drop(columns=['Sleep Disorder', 'Person ID'])
y = df['Sleep Disorder']

# --- PENTING: Simpan urutan kolom untuk digunakan di 'main.py' ---
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')
print(f"✅ Urutan kolom fitur disimpan: {feature_columns}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Ukuran data train: {X_train.shape}")
print(f"Ukuran data test : {X_test.shape}\n")

# =====================================================
# 5️⃣ Model 1: LightGBM
# =====================================================

print("--- TAHAP 5: TRAINING LIGHTGBM ---")
model_lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    objective='multiclass',  # Tentukan sebagai multiclass
    num_class=len(label_encoders['Sleep Disorder'].classes_) # Otomatis deteksi jumlah kelas
)
model_lgbm.fit(X_train, y_train)
y_pred_lgbm = model_lgbm.predict(X_test)

acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"✅ Akurasi LightGBM: {acc_lgbm:.4f}")
print(classification_report(y_test, y_pred_lgbm, target_names=label_encoders['Sleep Disorder'].classes_))

# =====================================================
# 6️⃣ Model 2: CatBoost
# =====================================================

print("\n--- TAHAP 6: TRAINING CATBOOST ---")
model_cat = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0,
    loss_function='MultiClass'  # Eksplisit untuk 3+ kelas
)
model_cat.fit(X_train, y_train)
y_pred_cat = model_cat.predict(X_test).flatten() # .flatten() untuk memastikan format array

acc_cat = accuracy_score(y_test, y_pred_cat)
print(f"✅ Akurasi CatBoost: {acc_cat:.4f}")
print(classification_report(y_test, y_pred_cat, target_names=label_encoders['Sleep Disorder'].classes_))

# =====================================================
# 7️⃣ Visualisasi Perbandingan Model
# =====================================================
print("\n--- TAHAP 7: VISUALISASI ---")
print("\n📊 Perbandingan Akurasi:")
print(f"LightGBM : {acc_lgbm:.4f}")
print(f"CatBoost : {acc_cat:.4f}")

# Dapatkan nama kelas dari encoder
class_names = label_encoders['Sleep Disorder'].classes_

fig, ax = plt.subplots(1, 2, figsize=(16, 6)) # figsize diubah agar muat 3x3
sns.heatmap(confusion_matrix(y_test, y_pred_lgbm), annot=True, fmt='d', cmap='Blues', ax=ax[0],
            xticklabels=class_names, yticklabels=class_names) # Menambahkan label
ax[0].set_title("Confusion Matrix - LightGBM")
ax[0].set_ylabel("Aktual")
ax[0].set_xlabel("Prediksi")

sns.heatmap(confusion_matrix(y_test, y_pred_cat), annot=True, fmt='d', cmap='Greens', ax=ax[1],
            xticklabels=class_names, yticklabels=class_names) # Menambahkan label
ax[1].set_title("Confusion Matrix - CatBoost")
ax[1].set_ylabel("Aktual")
ax[1].set_xlabel("Prediksi")

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png")  # disimpan sebagai file
print("📸 Confusion matrix disimpan sebagai 'confusion_matrix_comparison.png'.")

# =====================================================
# 8️⃣ Simulasi Prediksi Data Baru (Sesuai Kolom Asli)
# =====================================================

print("\n🔮 Prediksi Data Baru (Simulasi User Input)")

# Buat contoh input baru (data mentah/string)
new_data_dict = {
    'Gender': 'Male',
    'Age': 35,
    'Occupation': 'Nurse',
    'Sleep Duration': 6.5,
    'Quality of Sleep': 6,
    'Physical Activity Level': 40,
    'Stress Level': 6,
    'BMI Category': 'Overweight',
    'Blood Pressure': '120/80',
    'Heart Rate': 80,
    'Daily Steps': 7000
}

# Membuat DataFrame kosong dengan kolom yang benar
encoded_df = pd.DataFrame(columns=feature_columns)
# Menambahkan data baru ke DataFrame
encoded_df.loc[0] = new_data_dict

# Melakukan encoding pada data baru menggunakan encoder yang sudah disimpan
for col in label_encoders:
    if col in encoded_df.columns: # Hanya encode kolom yang ada di fitur
        val = encoded_df[col].iloc[0]
        try:
            # Mengubah nilai string/numerik menjadi label (angka)
            encoded_df[col] = label_encoders[col].transform([val])[0]
        except Exception as e:
            print(f"Error encoding kolom {col} dengan nilai '{val}': {e}")
            encoded_df[col] = np.nan # Jika error, set ke NaN (atau nilai default lain)

# Memastikan urutan kolom sesuai dengan saat training
final_df_to_predict = encoded_df[feature_columns] 

print("\nData baru setelah di-encode:")
print(final_df_to_predict.head())


# Prediksi
pred_lgbm = model_lgbm.predict(final_df_to_predict)[0]
pred_cat = model_cat.predict(final_df_to_predict)[0]

# Decode label (mengubah 0, 1, 2 kembali ke "Insomnia", "Sleep Apnea", "Tidak Ada")
hasil_lgbm = label_encoders['Sleep Disorder'].inverse_transform([pred_lgbm])[0]
hasil_cat = label_encoders['Sleep Disorder'].inverse_transform([pred_cat])[0]

print(f"\n💡 Hasil Prediksi LightGBM : {hasil_lgbm}")
print(f"💡 Hasil Prediksi CatBoost : {hasil_cat}")

if hasil_lgbm == hasil_cat:
    print(f"✅ Kedua model sepakat: kemungkinan besar adalah **{hasil_cat}**.")
else:
    print(f"⚠️ Hasil berbeda antara model. LightGBM → {hasil_lgbm}, CatBoost → {hasil_cat}")

# =====================================================
# 9️⃣ Simpan Model dan Encoder
# =====================================================

joblib.dump(model_lgbm, "model_lightgbm.pkl")
joblib.dump(model_cat, "model_catboost.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n💾 Model berhasil disimpan sebagai:")
print(" - model_lightgbm.pkl")
print(" - model_catboost.pkl")
print(" - label_encoders.pkl")
print("\n✅ Proses training selesai dengan sukses!")

