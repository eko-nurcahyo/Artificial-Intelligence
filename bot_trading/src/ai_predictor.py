import joblib
from catboost import CatBoostClassifier

class AIPredictor:
    def __init__(self, lgbm_model_path, catboost_model_path):
        # Load LightGBM model
        self.model_lgbm = joblib.load(lgbm_model_path)
        
        # Load CatBoost model
        self.model_catboost = CatBoostClassifier()
        self.model_catboost.load_model(catboost_model_path)

    def predict_direction(self, features):
        """
        Melakukan prediksi arah harga (BUY/SELL/HOLD) menggunakan ensemble LightGBM dan CatBoost.
        
        Parameter:
        features : pd.DataFrame or 2D-array
            Fitur input dari data teknikal terbaru, berformat dataframe atau numpy 2D-array.

        Return:
        final_pred : str
            Prediksi akhir arah pasar ("BUY", "SELL", atau "HOLD").
        confidence : float
            Confidence score tertinggi di antara dua model.
        """
        # Prediksi kelas dari masing-masing model
        pred_lgbm = self.model_lgbm.predict(features)[0]
        pred_cat = self.model_catboost.predict(features)[0]

        # Prediksi probabilitas/confidence dari masing-masing model
        conf_lgbm = max(self.model_lgbm.predict_proba(features)[0])
        conf_cat = max(self.model_catboost.predict_proba(features)[0])

        # Logika voting ensemble
        if pred_lgbm == pred_cat:
            final_pred = pred_lgbm
            confidence = max(conf_lgbm, conf_cat)
        else:
            # Jika beda hasil, pilih prediksi dari model dengan confidence tertinggi
            if conf_lgbm > conf_cat:
                final_pred = pred_lgbm
                confidence = conf_lgbm
            else:
                final_pred = pred_cat
                confidence = conf_cat

        return final_pred, confidence
