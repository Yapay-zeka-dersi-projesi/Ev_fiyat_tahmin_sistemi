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

    label, .stSelectbox label, .stNumberInput label, .stCheckbox label {
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
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "xgboost_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
SEHIR_MAP_PATH = os.path.join(BASE_DIR, "sehir_map.pkl")
DATA_PATH     = os.path.join(BASE_DIR, "clean_data.csv")

# ─────────────────────────  Encoding sabitleri  ─────────────────────────
YAS_MAP = {
    "0 (Yeni)":     0,
    "1-5 Yıl":      3,
    "5-10 Yıl":     7,
    "11-15 Yıl":    13,
    "16-20 Yıl":    18,
    "21 Ve Üzeri":  25,
}

KULLANIM_MAP = {
    "Boş":      0.0,
    "Kiracılı": 1.0,
}

ISITMA_TIPLERI = [
    "Doğalgaz Sobalı",
    "Güneş Enerjisi",
    "Isıtma Yok",
    "Jeotermal",
    "Kat Kaloriferi",
    "Klimalı",
    "Kombi Doğalgaz",
    "Merkezi (Pay Ölçer)",
    "Merkezi Doğalgaz",
    "Merkezi Kömür",
    "Sobalı",
    "Yerden Isıtma",
]

# ─────────────────────────  Model & Veri Yükleme  ─────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error(
        "⚠️ Model dosyası bulunamadı!\n\n"
        "Lütfen önce `model.ipynb` not defterini çalıştırarak modeli kaydedin."
    )
    st.stop()

if not os.path.exists(SEHIR_MAP_PATH):
    st.error(
        "⚠️ `sehir_map.pkl` dosyası bulunamadı!\n\n"
        "Lütfen `build_sehir_map.py` betiğini çalıştırın:\n\n"
        "```\npython build_sehir_map.py\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Model yükleniyor…")
def load_model():
    _model        = joblib.load(MODEL_PATH)
    _feature_cols = joblib.load(FEATURES_PATH)
    _sehir_map    = joblib.load(SEHIR_MAP_PATH)
    return _model, _feature_cols, _sehir_map


@st.cache_data(show_spinner=False)
def get_options(_data_path):
    df = pd.read_csv(_data_path)
    return {
        "net_m2_min":    int(df["Net_Metrekare"].min()),
        "net_m2_max":    int(df["Net_Metrekare"].max()),
        "brut_m2_min":   int(df["Brüt_Metrekare"].min()),
        "brut_m2_max":   min(int(df["Brüt_Metrekare"].max()), 2000),
        "oda_sayisi":    sorted(df["Oda_Sayısı"].unique().tolist()),
        "banyo_sayisi":  sorted(df["Banyo_Sayısı"].unique().tolist()),
        "kat_sayisi_min":int(df["Binanın_Kat_Sayısı"].min()),
        "kat_sayisi_max":int(df["Binanın_Kat_Sayısı"].max()),
    }


model, feature_cols, sehir_map = load_model()
opts = get_options(DATA_PATH)

# Şehir listesi: sehir_map'ten türetilir, capitalize ile görüntülenir
sehir_listesi = sorted(sehir_map.keys())

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

# ─────────────────────────  FORM — Bölüm 1: Metrekare & Yapı  ─────────────────────────
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
    # Oda sayısı float; "2+1" → 2.5 gibi göster
    oda_labels = []
    oda_values = []
    for v in opts["oda_sayisi"]:
        if v == int(v):
            oda_labels.append(f"{int(v)+1}+0" if False else f"{int(v)} Oda")
        else:
            tam = int(v)
            oda_labels.append(f"{tam}+1")
        oda_values.append(v)
    oda_idx = st.selectbox("Oda Sayısı", options=range(len(oda_labels)), format_func=lambda i: oda_labels[i], index=2)
    oda = oda_values[oda_idx]

with col4:
    banyo_idx = st.selectbox("Banyo Sayısı", options=range(len(opts["banyo_sayisi"])),
                             format_func=lambda i: f"{int(opts['banyo_sayisi'][i])} Banyo", index=0)
    banyo = opts["banyo_sayisi"][banyo_idx]

with col5:
    bina_kat = st.number_input(
        "Binanın Kat Sayısı",
        min_value=opts["kat_sayisi_min"],
        max_value=opts["kat_sayisi_max"],
        value=5,
        step=1,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  FORM — Bölüm 2: Konum & Daire Özellikleri  ─────────────────────────
st.markdown('<div class="form-card"><div class="section-title">📍 Konum & Daire Özellikleri</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    default_sehir_idx = sehir_listesi.index("istanbul") if "istanbul" in sehir_listesi else 0
    sehir_key = st.selectbox(
        "Şehir",
        options=sehir_listesi,
        index=default_sehir_idx,
        format_func=lambda s: s.capitalize(),
    )
with col7:
    bina_yasi_label = st.selectbox("Binanın Yaşı", options=list(YAS_MAP.keys()), index=0)

col8, col9 = st.columns(2)
with col8:
    isitma_tipi = st.selectbox("Isıtma Tipi", options=ISITMA_TIPLERI, index=6)  # Kombi Doğalgaz default
with col9:
    kullanim_label = st.selectbox("Kullanım Durumu", options=list(KULLANIM_MAP.keys()), index=0)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  FORM — Bölüm 3: Daire Tipi  ─────────────────────────
st.markdown('<div class="form-card"><div class="section-title">🏢 Daire Tipi</div>', unsafe_allow_html=True)

col10, col11, col12 = st.columns(3)
with col10:
    apartman_mi = st.checkbox("Apartman / Site Dairesi", value=True,
                              help="Villa veya müstakil ev değilse işaretleyin")
with col11:
    zemin_mi = st.checkbox("Zemin / Bahçe Katı", value=False,
                           help="Zemin veya bahçe katında ise işaretleyin")
with col12:
    cati_mi = st.checkbox("Çatı Katı", value=False,
                          help="Çatı dubleks veya penthouse ise işaretleyin")

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────  Tahmin  ─────────────────────────
if st.button("🔍  Fiyat Tahmin Et"):
    with st.spinner("Tahmin hesaplanıyor…"):

        # --- Şehir encode (target encoding ile uyumlu) ---
        sehir_encoded = sehir_map.get(sehir_key, sehir_map[list(sehir_map.keys())[0]])

        # --- Binanın Yaşı ---
        bina_yasi_val = float(YAS_MAP[bina_yasi_label])

        # --- Kullanım Durumu ---
        kullanim_val = KULLANIM_MAP[kullanim_label]

        # --- Isıtma Tipi one-hot ---
        # Modeldeki sütun adları: "Isıtma_Tipi_<tip>"; drop_first=True → ilk tip (Doğalgaz Sobalı) baseline
        isitma_one_hot = {}
        for tip in ISITMA_TIPLERI[1:]:   # drop_first: ilk tip çıkarıldı
            col_name = f"Isıtma_Tipi_{tip}"
            isitma_one_hot[col_name] = (isitma_tipi == tip)

        # --- Temel input dict ---
        input_dict = {
            "Net_Metrekare":      [float(net_m2)],
            "Brüt_Metrekare":     [float(brut_m2)],
            "Oda_Sayısı":         [float(oda)],
            "Binanın_Yaşı":       [bina_yasi_val],
            "Şehir":              [sehir_encoded],
            "Binanın_Kat_Sayısı": [float(bina_kat)],
            "Kullanım_Durumu":    [kullanim_val],
            "Banyo_Sayısı":       [float(banyo)],
            "Apartman_Mi":        [float(int(apartman_mi))],
            "Zemin_Mi":           [float(int(zemin_mi))],
            "Cati_Mi":            [float(int(cati_mi))],
        }

        # Isıtma one-hot sütunlarını ekle
        for k, v in isitma_one_hot.items():
            input_dict[k] = [v]

        input_df = pd.DataFrame(input_dict)

        # Modelin beklediği tüm sütunları ekle / sırala
        for fc in feature_cols:
            if fc not in input_df.columns:
                input_df[fc] = False
        input_df = input_df[feature_cols]

        predicted_price = model.predict(input_df)[0]

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
            <div class="info-chip">Şehir<span>{sehir_key.capitalize()}</span></div>
            <div class="info-chip">Net m²<span>{net_m2} m²</span></div>
            <div class="info-chip">Oda<span>{oda_labels[oda_idx]}</span></div>
            <div class="info-chip">Bina Yaşı<span>{bina_yasi_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
