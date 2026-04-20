# Kullanımı:
# streamlit run c:/Users/metin/Desktop/Ev_fiyat_tahmin_sistemi/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# ─────────────────────────  Sayfa ayarları  ─────────────────────────
st.set_page_config(
    page_title="Ev Fiyat Tahmin Sistemi",
    page_icon="🏠",
    layout="centered",
)

# ─────────────────────────  CSS  ─────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .header-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 32px 36px 24px 36px;
        margin-bottom: 28px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .header-card p {
        color: rgba(255,255,255,0.55);
        font-size: 0.95rem;
        margin: 0;
    }

    .form-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        backdrop-filter: blur(8px);
    }
    .section-title {
        color: #a78bfa;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .result-box {
        background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(59,130,246,0.20));
        border: 1px solid rgba(167,139,250,0.45);
        border-radius: 18px;
        padding: 36px;
        text-align: center;
        margin-top: 10px;
    }
    .result-label {
        color: rgba(255,255,255,0.60);
        font-size: 0.88rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .result-price {
        color: #a78bfa;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .result-sub {
        color: rgba(255,255,255,0.40);
        font-size: 0.80rem;
        margin-top: 8px;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: rgba(255,255,255,0.80) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    input[type="number"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 0;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    .info-row {
        display: flex;
        gap: 10px;
        margin-top: 18px;
    }
    .info-chip {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 6px 12px;
        color: rgba(255,255,255,0.55);
        font-size: 0.78rem;
        flex: 1;
        text-align: center;
    }
    .info-chip span {
        display: block;
        color: rgba(255,255,255,0.85);
        font-weight: 600;
        font-size: 0.88rem;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────  Dosya yolları  ─────────────────────────
BASE_DIR      = os.path.dirname(__file__)
DATA_PATH     = os.path.join(BASE_DIR, "clean_data.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "xgboost_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# ─────────────────────────  Model & Veri Yükleme  ─────────────────────────
# Model dosyası yoksa kullanıcıyı bilgilendir
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error(
        "⚠️ Model dosyası bulunamadı!\n\n"
        "Lütfen önce terminalde şunu çalıştırın:\n\n"
        "```\npython save_model.py\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Model yükleniyor…")
def load_model():
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


@st.cache_data(show_spinner=False)
def get_options():
    data = pd.read_csv(DATA_PATH)
    return {
        "sehirler":      sorted(data["Şehir"].unique().tolist()),
        "kat_listesi":   sorted(
            data["Bulunduğu_Kat"].unique().tolist(),
            key=lambda x: (not x[0].isdigit(), x),
        ),
        "bina_yasi":     data["Binanın_Yaşı"].unique().tolist(),
        "isitma":        sorted(data["Isıtma_Tipi"].unique().tolist()),
        "kullanim":      data["Kullanım_Durumu"].unique().tolist(),
        "oda_sayisi":    sorted(data["Oda_Sayısı"].unique().tolist()),
        "banyo_sayisi":  sorted(data["Banyo_Sayısı"].unique().tolist()),
        "net_m2_min":    int(data["Net_Metrekare"].min()),
        "net_m2_max":    int(data["Net_Metrekare"].max()),
        "brut_m2_min":   int(data["Brüt_Metrekare"].min()),
        "brut_m2_max":   int(data["Brüt_Metrekare"].max()),
        "kat_sayisi_min":int(data["Binanın_Kat_Sayısı"].min()),
        "kat_sayisi_max":int(data["Binanın_Kat_Sayısı"].max()),
    }


model, feature_cols = load_model()
opts = get_options()

# ─────────────────────────  Başlık  ─────────────────────────
st.markdown(
    """
    <div class="header-card">
        <h1>🏠 Ev Fiyat Tahmin Sistemi</h1>
        <p>XGBoost modeli ile ev özelliklerinizi girerek tahmini piyasa fiyatını öğrenin</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────  Form  ─────────────────────────
st.markdown('<div class="form-card"><div class="section-title">📐 Metrekare & Yapı Bilgileri</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    net_m2 = st.number_input(
        "Net Metrekare (m²)",
        min_value=opts["net_m2_min"],
        max_value=opts["net_m2_max"],
        value=100,
        step=5,
    )
with col2:
    brut_m2 = st.number_input(
        "Brüt Metrekare (m²)",
        min_value=opts["brut_m2_min"],
        max_value=opts["brut_m2_max"],
        value=120,
        step=5,
    )

col3, col4, col5 = st.columns(3)
with col3:
    oda = st.selectbox("Oda Sayısı", options=opts["oda_sayisi"], index=1)
with col4:
    banyo = st.selectbox("Banyo Sayısı", options=opts["banyo_sayisi"], index=0)
with col5:
    bina_kat = st.number_input(
        "Binanın Kat Sayısı",
        min_value=opts["kat_sayisi_min"],
        max_value=opts["kat_sayisi_max"],
        value=5,
        step=1,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="form-card"><div class="section-title">📍 Konum & Daire Özellikleri</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    default_sehir = opts["sehirler"].index("istanbul") if "istanbul" in opts["sehirler"] else 0
    sehir = st.selectbox("Şehir", options=opts["sehirler"], index=default_sehir)
with col7:
    bulundugu_kat = st.selectbox("Bulunduğu Kat", options=opts["kat_listesi"], index=0)

col8, col9 = st.columns(2)
with col8:
    bina_yasi = st.selectbox("Binanın Yaşı", options=opts["bina_yasi"], index=0)
with col9:
    isitma = st.selectbox("Isıtma Tipi", options=opts["isitma"], index=0)

kullanim = st.selectbox("Kullanım Durumu", options=opts["kullanim"], index=0)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  Tahmin  ─────────────────────────
if st.button("🔍  Fiyat Tahmin Et"):
    with st.spinner("Tahmin hesaplanıyor…"):
        # Kullanıcı girdisinden ham DataFrame satırı
        input_dict = {
            "Net_Metrekare":      [net_m2],
            "Brüt_Metrekare":     [float(brut_m2)],
            "Oda_Sayısı":         [float(oda)],
            "Bulunduğu_Kat":      [bulundugu_kat],
            "Binanın_Yaşı":       [bina_yasi],
            "Isıtma_Tipi":        [isitma],
            "Şehir":              [sehir],
            "Binanın_Kat_Sayısı": [bina_kat],
            "Kullanım_Durumu":    [kullanim],
            "Banyo_Sayısı":       [float(banyo)],
        }
        input_df = pd.DataFrame(input_dict)

        # get_dummies ile encode et
        kategorik = input_df.select_dtypes(include=["object"]).columns
        input_enc = pd.get_dummies(input_df, columns=kategorik, drop_first=True)

        # Modelin beklediği tüm sütunları ekle / sırala
        for col in feature_cols:
            if col not in input_enc.columns:
                input_enc[col] = False
        input_enc = input_enc[feature_cols]

        predicted_price = model.predict(input_enc)[0]

    formatted   = f"{predicted_price:,.0f} ₺"
    formatted_m = f"{predicted_price / 1_000_000:.2f} Milyon ₺"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Tahmini Satış Fiyatı</div>
            <div class="result-price">{formatted_m}</div>
            <div class="result-sub">{formatted}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-row">
            <div class="info-chip">Şehir<span>{sehir.capitalize()}</span></div>
            <div class="info-chip">Net m²<span>{net_m2} m²</span></div>
            <div class="info-chip">Oda<span>{oda}</span></div>
            <div class="info-chip">Banyo<span>{banyo}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
