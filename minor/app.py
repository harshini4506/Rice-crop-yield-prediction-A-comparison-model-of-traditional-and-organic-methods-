import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from PIL import Image
import io
from urllib.parse import quote
from pathlib import Path
from report_utils import generate_pdf

# ---------------------------------
# LANGUAGE SUPPORT
# ---------------------------------
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi"
}

TRANSLATIONS = {
    "en": {
        "app_title": "🌾 Rice Yield Optimizer",
        "app_subtitle": "AI-Powered Prediction & Organic Farming Optimization",
        "nav_input": "📋 Input Parameters",
        "nav_predictions": "📊 Yield Predictions",
        "nav_recommendations": "🌿 Recommendations",
        "nav_summary": "✅ Summary & Export",
        "nav_feedback": "📝 Feedback & Learn",
        "nav_progress": "Progress",
        "page_indicator": "Page",
        "of": "of",
        "prev_btn": "⬅️ Previous",
        "next_btn": "Next ➡️",
        "start_over": "🔄 Start Over",
        "step_1": "Step 1: Enter Your Farm Details",
        "farm_info": "🌱 Farm Information",
        "basic_farm": "Basic Farm Information",
        "rice_variety": "🍚 Rice Variety",
        "soil_texture": "🪨 Soil Texture",
        "irrigation_type": "💧 Irrigation Type",
        "climate": "🌡️ Climate",
        "climate_conditions": "Climate Conditions",
        "rainfall": "🌧️ Annual Rainfall (mm)",
        "temperature": "🌡️ Avg Temperature (°C)",
        "humidity": "💨 Humidity (%)",
        "sunshine": "☀️ Daily Sunshine Hours",
        "soil": "🧪 Soil",
        "soil_properties": "Soil Properties & Nutrients",
        "soil_ph": "pH Level",
        "organic_matter": "🌿 Organic Matter (%)",
        "nitrogen": "🟦 Nitrogen (mg/kg)",
        "phosphorus": "🟦 Phosphorus (mg/kg)",
        "potassium": "🟦 Potassium (mg/kg)",
        "management": "📊 Management",
        "farm_management": "Farm Management Practices",
        "irrigation_freq": "💧 Irrigation Frequency (times/season)",
        "planting_density": "📍 Planting Density (plants/m²)",
        "pest_index": "🐛 Pest Pressure Index (0-10)",
        "pest_guide": "Pest Severity Guide:",
        "image_upload": "📸 Upload Input Sheet Photo",
        "upload_instruction": "Or upload a photo of your data sheet",
        "upload_button": "📷 Upload Image",
        "processing": "Processing image...",
        "extracted_data": "Extracted Data from Image:",
        "yield_predictions": "📊 Yield Predictions",
        "traditional_yield": "🌾 Traditional Yield",
        "organic_yield": "🌱 Organic Optimized Yield",
        "improvement": "📈 Yield Improvement",
        "share_experience": "📝 Share Your Experience",
        "help_improve": "Help us learn from your implementation results",
        "implemented": "Have you applied the organic recommendations?",
        "timeline": "Timeline of implementation:",
        "feedback_summary": "📈 Your Feedback Summary",
        "thank_you": "Thank you for your valuable feedback!",
    },
    "es": {
        "app_title": "🌾 Optimizador de Rendimiento de Arroz",
        "app_subtitle": "Predicción Impulsada por IA y Optimización de Agricultura Orgánica",
        "nav_input": "📋 Parámetros de Entrada",
        "nav_predictions": "📊 Predicciones de Rendimiento",
        "nav_recommendations": "🌿 Recomendaciones",
        "nav_summary": "✅ Resumen y Exportación",
        "nav_feedback": "📝 Comentarios y Aprendizaje",
        "nav_progress": "Progreso",
        "page_indicator": "Página",
        "of": "de",
        "prev_btn": "⬅️ Anterior",
        "next_btn": "Siguiente ➡️",
        "start_over": "🔄 Comenzar de Nuevo",
        "step_1": "Paso 1: Ingrese los Detalles de su Granja",
        "farm_info": "🌱 Información de la Granja",
        "basic_farm": "Información Básica de la Granja",
        "rice_variety": "🍚 Variedad de Arroz",
        "soil_texture": "🪨 Textura del Suelo",
        "irrigation_type": "💧 Tipo de Riego",
        "climate": "🌡️ Clima",
        "climate_conditions": "Condiciones Climáticas",
        "rainfall": "🌧️ Lluvia Anual (mm)",
        "temperature": "🌡️ Temperatura Promedio (°C)",
        "humidity": "💨 Humedad (%)",
        "sunshine": "☀️ Horas de Sol Diarias",
        "soil": "🧪 Suelo",
        "soil_properties": "Propiedades del Suelo y Nutrientes",
        "soil_ph": "Nivel de pH",
        "organic_matter": "🌿 Materia Orgánica (%)",
        "nitrogen": "🟦 Nitrógeno (mg/kg)",
        "phosphorus": "🟦 Fósforo (mg/kg)",
        "potassium": "🟦 Potasio (mg/kg)",
        "management": "📊 Gestión",
        "farm_management": "Prácticas de Gestión de Granjas",
        "irrigation_freq": "💧 Frecuencia de Riego (veces/temporada)",
        "planting_density": "📍 Densidad de Siembra (plantas/m²)",
        "pest_index": "🐛 Índice de Presión de Plagas (0-10)",
        "pest_guide": "Guía de Severidad de Plagas:",
        "image_upload": "📸 Cargar Foto de Hoja de Entrada",
        "upload_instruction": "O cargue una foto de su hoja de datos",
        "upload_button": "📷 Cargar Imagen",
        "processing": "Procesando imagen...",
        "extracted_data": "Datos Extraídos de la Imagen:",
        "yield_predictions": "📊 Predicciones de Rendimiento",
        "traditional_yield": "🌾 Rendimiento Tradicional",
        "organic_yield": "🌱 Rendimiento Orgánico Optimizado",
        "improvement": "📈 Mejora de Rendimiento",
        "share_experience": "📝 Comparta su Experiencia",
        "help_improve": "Ayúdenos a aprender de los resultados de su implementación",
        "implemented": "¿Ha aplicado las recomendaciones orgánicas?",
        "timeline": "Cronología de implementación:",
        "feedback_summary": "📈 Resumen de sus Comentarios",
        "thank_you": "¡Gracias por sus valiosos comentarios!",
    },
    "fr": {
        "app_title": "🌾 Optimisateur de Rendement du Riz",
        "app_subtitle": "Prédiction Alimentée par l'IA et Optimisation de l'Agriculture Biologique",
        "nav_input": "📋 Paramètres d'Entrée",
        "nav_predictions": "📊 Prédictions de Rendement",
        "nav_recommendations": "🌿 Recommandations",
        "nav_summary": "✅ Résumé et Exportation",
        "nav_feedback": "📝 Commentaires et Apprentissage",
        "nav_progress": "Progrès",
        "page_indicator": "Page",
        "of": "sur",
        "prev_btn": "⬅️ Précédent",
        "next_btn": "Suivant ➡️",
        "start_over": "🔄 Recommencer",
        "step_1": "Étape 1: Entrez les Détails de Votre Ferme",
        "farm_info": "🌱 Informations de la Ferme",
        "basic_farm": "Informations de Base de la Ferme",
        "rice_variety": "🍚 Variété de Riz",
        "soil_texture": "🪨 Texture du Sol",
        "irrigation_type": "💧 Type d'Irrigation",
        "climate": "🌡️ Climat",
        "climate_conditions": "Conditions Climatiques",
        "rainfall": "🌧️ Précipitations Annuelles (mm)",
        "temperature": "🌡️ Température Moyenne (°C)",
        "humidity": "💨 Humidité (%)",
        "sunshine": "☀️ Heures d'Ensoleillement Quotidien",
        "soil": "🧪 Sol",
        "soil_properties": "Propriétés du Sol et Éléments Nutritifs",
        "soil_ph": "Niveau de pH",
        "organic_matter": "🌿 Matière Organique (%)",
        "nitrogen": "🟦 Azote (mg/kg)",
        "phosphorus": "🟦 Phosphore (mg/kg)",
        "potassium": "🟦 Potassium (mg/kg)",
        "management": "📊 Gestion",
        "farm_management": "Pratiques de Gestion Agricole",
        "irrigation_freq": "💧 Fréquence d'Irrigation (fois/saison)",
        "planting_density": "📍 Densité de Plantation (plants/m²)",
        "pest_index": "🐛 Indice de Pression des Ravageurs (0-10)",
        "pest_guide": "Guide de Gravité des Ravageurs:",
        "image_upload": "📸 Charger la Photo de la Feuille d'Entrée",
        "upload_instruction": "Ou téléchargez une photo de votre feuille de données",
        "upload_button": "📷 Télécharger l'Image",
        "processing": "Traitement de l'image...",
        "extracted_data": "Données Extraites de l'Image:",
        "yield_predictions": "📊 Prédictions de Rendement",
        "traditional_yield": "🌾 Rendement Traditionnel",
        "organic_yield": "🌱 Rendement Organique Optimisé",
        "improvement": "📈 Amélioration du Rendement",
        "share_experience": "📝 Partagez Votre Expérience",
        "help_improve": "Aidez-nous à apprendre des résultats de votre mise en œuvre",
        "implemented": "Avez-vous appliqué les recommandations biologiques?",
        "timeline": "Calendrier de mise en œuvre:",
        "feedback_summary": "📈 Résumé de Vos Commentaires",
        "thank_you": "Merci pour vos précieux commentaires!",
    },
    "hi": {
        "app_title": "🌾 चावल की पैदावार अनुकूलक",
        "app_subtitle": "AI-संचालित भविष्यवाणी और जैविक खेती अनुकूलन",
        "nav_input": "📋 इनपुट पैरामीटर",
        "nav_predictions": "📊 पैदावार भविष्यवाणी",
        "nav_recommendations": "🌿 सिफारिशें",
        "nav_summary": "✅ सारांश और निर्यात",
        "nav_feedback": "📝 प्रतिक्रिया और सीखना",
        "nav_progress": "प्रगति",
        "page_indicator": "पृष्ठ",
        "of": "का",
        "prev_btn": "⬅️ पिछला",
        "next_btn": "अगला ➡️",
        "start_over": "🔄 फिर से शुरू करें",
        "step_1": "चरण 1: अपने खेत का विवरण दर्ज करें",
        "farm_info": "🌱 खेत की जानकारी",
        "basic_farm": "खेत की बुनियादी जानकारी",
        "rice_variety": "🍚 चावल की किस्म",
        "soil_texture": "🪨 मिट्टी की बनावट",
        "irrigation_type": "💧 सिंचाई का प्रकार",
        "climate": "🌡️ जलवायु",
        "climate_conditions": "जलवायु की स्थिति",
        "rainfall": "🌧️ वार्षिक वर्षा (मिमी)",
        "temperature": "🌡️ औसत तापमान (°C)",
        "humidity": "💨 आर्द्रता (%)",
        "sunshine": "☀️ दैनिक धूप के घंटे",
        "soil": "🧪 मिट्टी",
        "soil_properties": "मिट्टी के गुण और पोषक तत्व",
        "soil_ph": "pH स्तर",
        "organic_matter": "🌿 जैविक पदार्थ (%)",
        "nitrogen": "🟦 नाइट्रोजन (mg/kg)",
        "phosphorus": "🟦 फॉस्फोरस (mg/kg)",
        "potassium": "🟦 पोटेशियम (mg/kg)",
        "management": "📊 प्रबंधन",
        "farm_management": "खेत प्रबंधन प्रथाएं",
        "irrigation_freq": "💧 सिंचाई आवृत्ति (बार/मौसम)",
        "planting_density": "📍 रोपण घनत्व (पौधे/m²)",
        "pest_index": "🐛 कीट दबाव सूचकांक (0-10)",
        "pest_guide": "कीट गंभीरता गाइड:",
        "image_upload": "📸 इनपुट शीट फ़ोटो अपलोड करें",
        "upload_instruction": "या अपनी डेटा शीट की तस्वीर अपलोड करें",
        "upload_button": "📷 छवि अपलोड करें",
        "processing": "छवि को संसाधित कर रहे हैं...",
        "extracted_data": "छवि से निकाला गया डेटा:",
        "yield_predictions": "📊 पैदावार भविष्यवाणी",
        "traditional_yield": "🌾 पारंपरिक पैदावार",
        "organic_yield": "🌱 जैविक अनुकूलित पैदावार",
        "improvement": "📈 पैदावार सुधार",
        "share_experience": "📝 अपना अनुभव साझा करें",
        "help_improve": "अपने कार्यान्वयन परिणामों से हमें सीखने में मदद करें",
        "implemented": "क्या आपने जैविक सिफारिशें लागू की हैं?",
        "timeline": "कार्यान्वयन समयरेखा:",
        "feedback_summary": "📈 आपकी प्रतिक्रिया का सारांश",
        "thank_you": "आपकी मूल्यवान प्रतिक्रिया के लिए धन्यवाद!",
    }
}

def t(key):
    """Translate key based on selected language"""
    lang = st.session_state.get('language', 'en')
    return TRANSLATIONS.get(lang, {}).get(key, key)

# ---------------------------------
# STREAMLIT CONFIG & STYLING
# ---------------------------------
st.set_page_config(
    page_title="Rice Yield Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "🌾 Rice Yield Prediction & Optimization System"}
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    :root {
        --primary-green: #2ecc71;
        --dark-green: #27ae60;
        --light-green: #d5f4e6;
        --accent-orange: #f39c12;
        --dark-bg: #0f3460;
        --text-dark: #1a1a1a;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 40px 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(46, 204, 113, 0.3);
    }
    
    .main-header h1 {
        font-size: 3em;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        font-size: 1.2em;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 1.3em;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(39, 174, 96, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d5f4e6 0%, #c8e6c9 100%);
        border-left: 5px solid #27ae60;
        padding: 20px;
        border-radius: 8px;
        color: #1b5e20;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .comparison-highlight {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #f39c12;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #3e2723;
    }

    .comparison-highlight * {
        color: #3e2723 !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f0f7f4 0%, #ffffff 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(46, 204, 113, 0.1);
    }

    /* Keep metric text readable on light cards even when app theme is dark */
    div[data-testid="stMetric"] {
        color: #0f3d2f !important;
    }

    div[data-testid="stMetric"],
    div[data-testid="stMetric"] *,
    div[data-testid="stMetric"] p,
    div[data-testid="stMetric"] span,
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div,
    div[data-testid="stMetric"] small,
    div[data-testid="stMetric"] strong,
    div[data-testid="stMetric"] a {
        color: #123b2d !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #123b2d !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] span {
        color: #123b2d !important;
        opacity: 1 !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] p,
    div[data-testid="stMetricValue"] span {
        color: #0f3d2f !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #0f3d2f !important;
    }

    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] *,
    div[data-testid="stMetricDelta"] > div,
    div[data-testid="stMetricDelta"] p,
    div[data-testid="stMetricDelta"] span {
        color: #1b5e20 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #1b5e20 !important;
    }

    .success-box,
    .success-box * {
        color: #1b5e20 !important;
    }
    
    .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        gap: 10px;
    }
    
    .page-indicator {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.05) 100%);
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.1em;
        font-weight: bold;
        color: #27ae60;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 1

if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'rice_variety': 'Basmati',
        'soil_texture': 'Loamy',
        'irrigation_type': 'Drip Irrigation',
        'rainfall': 900,
        'temperature': 28,
        'humidity': 70,
        'sunshine': 8.0,
        'soil_ph': 6.5,
        'organic_matter': 2.0,
        'nitrogen': 60,
        'phosphorus': 40,
        'potassium': 50,
        'irrigation_freq': 4,
        'pest_index': 3.0,
        'planting_density': 25
    }

if 'traditional_yield' not in st.session_state:
    st.session_state.traditional_yield = 0.0

if 'organic_yield' not in st.session_state:
    st.session_state.organic_yield = 0.0

if 'improvement' not in st.session_state:
    st.session_state.improvement = 0.0

if 'percentage_increase' not in st.session_state:
    st.session_state.percentage_increase = 0.0

if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {
        'took_recommendation': None,
        'implementation_timeline': 'Not started',
        'profit_observed': '',
        'drawbacks': '',
        'recommendations_feedback': '',
        'overall_satisfaction': 5,
        'yield_change': 0.0,
        'would_recommend': None
    }

# ---------------------------------
# LOAD MODEL & DATA
# ---------------------------------
try:
    model_path = Path(__file__).resolve().parent / "xgboost_yield_model.pkl"
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'xgboost_yield_model.pkl' exists.")
    st.stop()

# Define sample data for dropdowns
rice_varieties = ["Basmati", "Jasmine", "Arborio", "Sushi Rice", "Long Grain", "Short Grain", "Medium Grain"]
soil_textures = ["Sandy", "Loamy", "Clay", "Silty", "Sandy Loam", "Clay Loam"]
irrigation_types = ["Flood Irrigation", "Drip Irrigation", "Sprinkler", "Basin", "Furrow"]

# ---------------------------------
# PAGE 1: INPUT PARAMETERS (IMPROVED)
# ---------------------------------
def page_1_inputs():
    st.markdown(f"""
        <div class="main-header">
            <h1>{t("app_title")}</h1>
            <p>{t("app_subtitle")}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="section-header">{t("step_1")}</div>', unsafe_allow_html=True)
    
    # Use tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([t("farm_info"), t("climate"), t("soil"), t("management"), "📸 Auto-Fill"])
    
    with tab1:
        st.markdown(f"### {t('basic_farm')}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.form_data['rice_variety'] = st.selectbox(
                t("rice_variety"),
                rice_varieties,
                index=rice_varieties.index(st.session_state.form_data['rice_variety']),
                help="Different rice varieties have different yield potentials",
                key="rice_variety_select"
            )
        
        with col2:
            st.session_state.form_data['soil_texture'] = st.selectbox(
                t("soil_texture"),
                soil_textures,
                index=soil_textures.index(st.session_state.form_data['soil_texture']),
                help="Sandy, loamy, or clay soil affects nutrient retention",
                key="soil_texture_select"
            )
        
        st.session_state.form_data['irrigation_type'] = st.selectbox(
            t("irrigation_type"),
            irrigation_types,
            index=irrigation_types.index(st.session_state.form_data['irrigation_type']),
            help="Select your primary irrigation method",
            key="irrigation_type_select"
        )
        
        st.info("💡 These basic parameters help establish your farm's baseline conditions.")
    
    with tab2:
        st.markdown(f"### {t('climate_conditions')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{t('rainfall')} & {t('temperature')}**")
            st.session_state.form_data['rainfall'] = st.slider(
                t("rainfall"), 
                300, 2000, 
                st.session_state.form_data['rainfall'], 
                step=50,
                help="Total annual rainfall in millimeters"
            )
            st.session_state.form_data['temperature'] = st.slider(
                t("temperature"), 
                15, 45, 
                st.session_state.form_data['temperature'], 
                step=1,
                help="Average growing season temperature"
            )
        
        with col2:
            st.markdown(f"**{t('humidity')} & {t('sunshine')}**")
            st.session_state.form_data['humidity'] = st.slider(
                t("humidity"), 
                30, 100, 
                st.session_state.form_data['humidity'], 
                step=5,
                help="Average relative humidity percentage"
            )
            st.session_state.form_data['sunshine'] = st.slider(
                t("sunshine"), 
                4.0, 12.0, 
                st.session_state.form_data['sunshine'], 
                step=0.5,
                help="Average daily sunshine hours"
            )
        
        st.success("✓ Climate data helps predict optimal growing conditions")
    
    with tab3:
        st.markdown(f"### {t('soil_properties')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{t('soil_ph')} & {t('organic_matter')}**")
            st.session_state.form_data['soil_ph'] = st.slider(
                t("soil_ph"), 
                4.0, 9.0, 
                st.session_state.form_data['soil_ph'], 
                step=0.1,
                help="Soil pH affects nutrient availability (6.5 is ideal for rice)"
            )
            st.session_state.form_data['organic_matter'] = st.slider(
                t("organic_matter"), 
                0.5, 5.0, 
                st.session_state.form_data['organic_matter'], 
                step=0.1,
                help="Percentage of organic matter in soil"
            )
        
        with col2:
            st.markdown(f"**{t('nitrogen')}, {t('phosphorus')}, {t('potassium')}**")
            st.session_state.form_data['nitrogen'] = st.slider(
                t("nitrogen"), 
                0, 150, 
                st.session_state.form_data['nitrogen'], 
                step=5,
                help="Nitrogen content in soil"
            )
            st.session_state.form_data['phosphorus'] = st.slider(
                t("phosphorus"), 
                0, 100, 
                st.session_state.form_data['phosphorus'], 
                step=5,
                help="Phosphorus content in soil"
            )
            st.session_state.form_data['potassium'] = st.slider(
                t("potassium"), 
                0, 120, 
                st.session_state.form_data['potassium'], 
                step=5,
                help="Potassium content in soil"
            )
        
        st.info("📊 Soil testing recommended for accurate nutrient levels")
    
    with tab4:
        st.markdown(f"### {t('farm_management')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.form_data['irrigation_freq'] = st.slider(
                t("irrigation_freq"), 
                1, 10, 
                st.session_state.form_data['irrigation_freq'], 
                step=1,
                help="How many times do you irrigate per season?"
            )
            st.session_state.form_data['planting_density'] = st.slider(
                t("planting_density"), 
                10, 40, 
                st.session_state.form_data['planting_density'], 
                step=1,
                help="Number of rice plants per square meter"
            )
        
        with col2:
            st.session_state.form_data['pest_index'] = st.slider(
                t("pest_index"), 
                0.0, 10.0, 
                st.session_state.form_data['pest_index'], 
                step=0.5,
                help="Rate pest pressure: 0=none, 10=severe"
            )
            
            st.markdown(f"**{t('pest_guide')}**")
            st.markdown("""
            - 0-2: Very low pest pressure
            - 3-5: Moderate pest pressure  
            - 6-8: High pest pressure
            - 9-10: Severe infestation risk
            """)
        
        st.success("✓ Management practices significantly impact final yield")
    
    with tab5:
        st.markdown(f"### {t('image_upload')}")
        st.markdown(t("upload_instruction"))
        
        uploaded_file = st.file_uploader(
            t("upload_button"),
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="farm_data_image"
        )
        
        if uploaded_file is not None:
            try:
                st.info(t("processing"))
                
                # Load and display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Data Sheet", use_column_width=True)
                
                # Try to extract text using basic vision (placeholder for OCR)
                st.markdown(f"### {t('extracted_data')}")
                
                # Since we don't have pytesseract, we'll show a placeholder
                st.warning("🔍 Advanced Image Recognition: For production, integrate with OCR (Tesseract/EasyOCR)")
                
                # Allow manual entry of extracted values
                st.markdown("**Manual Entry from Image:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    extracted_rainfall = st.number_input("Rainfall from image (mm)", 300, 2000, 900, key="extracted_rainfall")
                    extracted_temp = st.number_input("Temperature from image (°C)", 15, 45, 28, key="extracted_temp")
                    extracted_nitrogen = st.number_input("Nitrogen from image (mg/kg)", 0, 150, 60, key="extracted_nitrogen")
                
                with col2:
                    extracted_phosphorus = st.number_input("Phosphorus from image (mg/kg)", 0, 100, 40, key="extracted_phosphorus")
                    extracted_potassium = st.number_input("Potassium from image (mg/kg)", 0, 120, 50, key="extracted_potassium")
                    extracted_ph = st.number_input("Soil pH from image", 4.0, 9.0, 6.5, step=0.1, key="extracted_ph")
                
                if st.button("✅ Apply Extracted Data", key="apply_extracted_data"):
                    st.session_state.form_data['rainfall'] = int(extracted_rainfall)
                    st.session_state.form_data['temperature'] = int(extracted_temp)
                    st.session_state.form_data['nitrogen'] = int(extracted_nitrogen)
                    st.session_state.form_data['phosphorus'] = int(extracted_phosphorus)
                    st.session_state.form_data['potassium'] = int(extracted_potassium)
                    st.session_state.form_data['soil_ph'] = extracted_ph
                    st.success("✅ Data applied successfully! Go to other tabs to adjust other parameters.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
# ---------------------------------
# PAGE 2: YIELD PREDICTIONS
# ---------------------------------
def page_2_predictions():
    st.markdown("""
        <div class="main-header">
            <h1>📊 Yield Predictions</h1>
            <p>Traditional vs Organic Optimization Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create predictions
    user_input = pd.DataFrame([{
        "Rice_Variety": st.session_state.form_data['rice_variety'],
        "Soil_Texture": st.session_state.form_data['soil_texture'],
        "Irrigation_Type": st.session_state.form_data['irrigation_type'],
        "Rainfall_mm": st.session_state.form_data['rainfall'],
        "Temperature_C": st.session_state.form_data['temperature'],
        "Humidity_%": st.session_state.form_data['humidity'],
        "Sunshine_Hours": st.session_state.form_data['sunshine'],
        "Soil_pH": st.session_state.form_data['soil_ph'],
        "Soil_Organic_Matter_%": st.session_state.form_data['organic_matter'],
        "Nitrogen": st.session_state.form_data['nitrogen'],
        "Phosphorus": st.session_state.form_data['phosphorus'],
        "Potassium": st.session_state.form_data['potassium'],
        "Irrigation_Frequency": st.session_state.form_data['irrigation_freq'],
        "Pest_Severity_Index": st.session_state.form_data['pest_index'],
        "Planting_Density": st.session_state.form_data['planting_density']
    }])
    
    traditional_yield = model.predict(user_input)[0]
    
    # Organic optimization
    optimized_input = user_input.copy()
    optimized_input["Nitrogen"] *= 1.15
    optimized_input["Phosphorus"] *= 1.10
    optimized_input["Potassium"] *= 1.12
    optimized_input["Soil_Organic_Matter_%"] *= 1.20
    
    organic_yield = model.predict(optimized_input)[0]
    improvement = organic_yield - traditional_yield
    percentage_increase = (improvement / traditional_yield) * 100
    
    st.markdown('<div class="section-header">📈 Yield Comparison</div>', unsafe_allow_html=True)
    
    # Metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            "🌾 Traditional Yield",
            f"{traditional_yield:.4f} tons/ha",
            help="Predicted yield with traditional farming"
        )
    
    with metric_col2:
        st.metric(
            "🌱 Organic Optimized Yield",
            f"{organic_yield:.4f} tons/ha",
            help="Predicted yield with organic optimization"
        )
    
    with metric_col3:
        st.metric(
            "📈 Yield Improvement",
            f"{improvement:.4f} tons/ha",
            f"{percentage_increase:.4f}%",
            delta_color="off"
        )
    
    # Improvement box
    st.markdown(f"""
        <div class="success-box">
            ✨ With organic optimization, you can achieve <b>{percentage_increase:.4f}%</b> higher yield!
            <br>That's an additional <b>{improvement:.4f} tons/ha</b> of rice production.
        </div>
    """, unsafe_allow_html=True)
    
    # Optimization parameters
    st.markdown('<div class="section-header">🔧 Applied Optimizations</div>', unsafe_allow_html=True)
    
    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
    with opt_col1:
        st.metric("🟦 Nitrogen Boost", "×1.15", "+15%")
    with opt_col2:
        st.metric("🟦 Phosphorus Boost", "×1.10", "+10%")
    with opt_col3:
        st.metric("🟦 Potassium Boost", "×1.12", "+12%")
    with opt_col4:
        st.metric("🌿 Organic Matter Boost", "×1.20", "+20%")
    
    # Comparison table
    st.markdown("### 📋 Detailed Comparison")
    comparison_df = pd.DataFrame({
        "🌾 Category": ["Traditional Farming", "🌱 Organic Optimized"],
        "Predicted Yield (tons/ha)": [round(traditional_yield, 4), round(organic_yield, 4)],
        "Nutrient Level": ["Standard", "Enhanced"],
        "Sustainability": ["Conventional", "Organic"]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown('<div class="section-header">📈 Visual Analysis</div>', unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Traditional\nFarming", "Organic\nOptimized"]
        yields = [traditional_yield, organic_yield]
        colors = ['#3498db', '#2ecc71']
        bars = ax.bar(categories, yields, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        for bar, yield_val in zip(bars, yields):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{yield_val:.4f}\ntons/ha',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Yield (tons/ha)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Farming Method', fontsize=12, fontweight='bold')
        ax.set_title('Yield Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(yields) * 1.15)
        plt.tight_layout()
        st.pyplot(fig)
    
    with viz_col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Traditional", "Organic"]
        values = [traditional_yield, organic_yield]
        colors_pie = ['#3498db', '#2ecc71']
        
        x_pos = np.arange(len(categories))
        bars = ax.barh(x_pos, values, color=colors_pie, alpha=0.8, edgecolor='black', linewidth=2)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            label = f'{val:.4f} tons/ha'
            if i == 1:
                label += f'\n({percentage_increase:+.4f}%)'
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'  {label}',
                    ha='left', va='center', fontweight='bold', fontsize=11)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_xlabel('Yield (tons/ha)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Store predictions in session state for next page
    st.session_state.traditional_yield = traditional_yield
    st.session_state.organic_yield = organic_yield
    st.session_state.improvement = improvement
    st.session_state.percentage_increase = percentage_increase

# ---------------------------------
# PAGE 3: ORGANIC RECOMMENDATIONS
# ---------------------------------
def page_3_recommendations():
    st.markdown("""
        <div class="main-header">
            <h1>🌿 Organic Recommendations</h1>
            <p>Component Mix & Implementation Guide</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🌿 Organic Component Recommendations</div>', unsafe_allow_html=True)
    
    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
    
    components_list = [
        ("🪱 Vermicompost", 30, "#ff6b6b"),
        ("🌾 Green Manure", 25, "#4ecdc4"),
        ("🦴 Bone Meal", 20, "#45b7d1"),
        ("🧬 Bio-Fertilizer", 25, "#96ceb4")
    ]
    
    for comp_col, (name, percentage, color) in zip(
        [comp_col1, comp_col2, comp_col3, comp_col4],
        components_list
    ):
        with comp_col:
            st.metric(name, f"{percentage}%")
    
    st.markdown("""
        <div class="success-box">
            💡 <b>Application Tip:</b> Mix the organic components in the proportions shown above. 
            This balanced composition ensures optimal nutrient delivery and soil health improvement 
            for achieving the predicted organic yield.
        </div>
    """, unsafe_allow_html=True)
    
    # Component pie chart
    st.markdown('<div class="section-header">📊 Component Mix Breakdown</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    components_names = [c[0].replace('🪱 ', '').replace('🌾 ', '').replace('🦴 ', '').replace('🧬 ', '') 
                         for c in components_list]
    percentages = [c[1] for c in components_list]
    colors = [c[2] for c in components_list]
    
    wedges, texts, autotexts = ax.pie(percentages, labels=components_names, autopct='%1.0f%%',
                                        colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'},
                                        explode=(0.05, 0.05, 0.05, 0.05))
    
    ax.set_title('Recommended Organic Component Mix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Implementation guide
    st.markdown('<div class="section-header">📋 Implementation Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### Phase 1: Preparation (Week 1-2)
    - 🔍 Conduct soil test to verify pH and nutrient levels
    - 📦 Source organic components from certified suppliers
    - 🛠️ Prepare composting area if needed
    
    #### Phase 2: Application (Week 3-4)
    - 🪱 Apply 30% Vermicompost - enhances microbial activity
    - 🌾 Incorporate 25% Green Manure - adds nitrogen naturally
    - 🦴 Mix 20% Bone Meal - provides phosphorus and calcium
    - 🧬 Add 25% Bio-Fertilizer Compost - balanced nutrients
    
    #### Phase 3: Maintenance (Week 5+)
    - 💧 Optimize irrigation schedule to {irrigation_freq} times
    - 🐛 Monitor pest levels weekly
    - 📊 Track soil health every 2 weeks
    - 📈 Monitor yield development
    """.format(irrigation_freq=st.session_state.form_data['irrigation_freq']))
    
    st.markdown("""
        <div class="comparison-highlight">
            ⚡ <b>Quick Tips for Success:</b><br>
            • Start with small test plots before full-scale application<br>
            • Maintain soil moisture at 70% capacity<br>
            • Avoid pesticides during organic transition<br>
            • Keep detailed records of application dates and quantities
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# PAGE 4: SUMMARY & EXPORT
# ---------------------------------
def page_4_summary():
    st.markdown("""
        <div class="main-header">
            <h1>✅ Summary & Export</h1>
            <p>Complete Analysis Report</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📋 Your Farming Profile</div>', unsafe_allow_html=True)
    
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        st.markdown("""
        **🌾 Farm Details:**
        - 🍚 Rice Variety: """ + st.session_state.form_data['rice_variety'] + """
        - 🪨 Soil Texture: """ + st.session_state.form_data['soil_texture'] + """
        - 💧 Irrigation Type: """ + st.session_state.form_data['irrigation_type'] + """
        - 🌡️ Temperature: """ + str(st.session_state.form_data['temperature']) + """°C
        - 💨 Humidity: """ + str(st.session_state.form_data['humidity']) + """%
        """)
    
    with profile_col2:
        st.markdown("""
        **🧪 Soil Conditions:**
        - pH Level: """ + str(st.session_state.form_data['soil_ph']) + """
        - Organic Matter: """ + str(st.session_state.form_data['organic_matter']) + """%
        - Nitrogen: """ + str(st.session_state.form_data['nitrogen']) + """
        - Phosphorus: """ + str(st.session_state.form_data['phosphorus']) + """
        - Potassium: """ + str(st.session_state.form_data['potassium']) + """
        """)
    
    st.markdown('<div class="section-header">📊 Final Predictions</div>', unsafe_allow_html=True)
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            "🌾 Traditional Yield",
            f"{st.session_state.traditional_yield:.4f} tons/ha"
        )
    
    with result_col2:
        st.metric(
            "🌱 Organic Optimized Yield",
            f"{st.session_state.organic_yield:.4f} tons/ha"
        )
    
    with result_col3:
        st.metric(
            "📈 Total Improvement",
            f"{st.session_state.percentage_increase:.4f}%"
        )
    
    st.markdown(f"""
        <div class="success-box">
            🎯 <b>Expected Results:</b><br>
            Yield Gain: {st.session_state.improvement:+.4f} tons/ha<br>
            Percentage Increase: {st.session_state.percentage_increase:+.4f}%<br>
            Projected Total Yield: {st.session_state.organic_yield:.4f} tons/ha
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">✅ Next Steps & Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **📊 Review Analysis** - Verify all predictions match your farm conditions
    2. **🛒 Source Materials** - Order verified organic components
    3. **📅 Plan Schedule** - Prepare implementation timeline
    4. **📸 Document Baseline** - Take photos before applying changes
    5. **📈 Monitor Progress** - Track yield development throughout season
    6. **📊 Analyze Results** - Compare actual vs predicted yields
    """)
    
    # Export section
    st.markdown('<div class="section-header">💾 Export Report</div>', unsafe_allow_html=True)
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        try:
            recommended_fertilizer = round(st.session_state.form_data.get('nitrogen', 0) * 1.15, 2)
            pdf_path = generate_pdf(
                st.session_state.traditional_yield,
                st.session_state.organic_yield,
                st.session_state.improvement,
                recommended_fertilizer
            )

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            st.download_button(
                "📥 Download PDF Report",
                data=pdf_bytes,
                file_name="Yield_Report.pdf",
                mime="application/pdf",
                key="pdf_download_btn"
            )
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")
    
    with export_col2:
        if st.button("📤 Share Results", key="share_btn"):
            share_message = (
                "Rice Yield Optimization Results\n"
                f"Farm: {st.session_state.form_data['rice_variety']} | {st.session_state.form_data['soil_texture']}\n"
                f"Traditional Yield: {st.session_state.traditional_yield:.4f} tons/ha\n"
                f"Organic Optimized Yield: {st.session_state.organic_yield:.4f} tons/ha\n"
                f"Improvement: {st.session_state.improvement:+.4f} tons/ha "
                f"({st.session_state.percentage_increase:+.4f}%)\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            encoded_subject = quote("Rice Yield Optimization Results")
            encoded_body = quote(share_message)
            encoded_whatsapp = quote(share_message)

            st.success("✅ Share content prepared. Copy the text or use quick share links below.")
            st.text_area(
                "Share summary",
                value=share_message,
                height=180,
                key="share_summary_text",
                help="Copy this text and paste it in chat, email, or your team channel"
            )

            share_link_col1, share_link_col2 = st.columns(2)
            with share_link_col1:
                st.markdown(
                    f"[📧 Share via Email](mailto:?subject={encoded_subject}&body={encoded_body})"
                )
            with share_link_col2:
                st.markdown(
                    f"[💬 Share via WhatsApp](https://wa.me/?text={encoded_whatsapp})"
                )

            st.download_button(
                "📄 Download Share Summary (.txt)",
                data=share_message,
                file_name="rice_yield_share_summary.txt",
                mime="text/plain",
                key="download_share_summary"
            )
    
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 40px;'>
            🌾 <b>Rice Yield Optimization System</b> | Powered by XGBoost ML Model<br>
            💚 Supporting sustainable and productive agriculture<br>
            <small>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</small>
        </div>
    """, unsafe_allow_html=True)

# =================================
# PAGE 5: USER FEEDBACK & LEARNING
# =================================
def page_5_feedback():
    st.markdown("""
        <div class="main-header">
            <h1>📝 Share Your Experience</h1>
            <p>Help us learn from your implementation results</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎯 Feedback & Results Collection</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This feedback helps us improve our AI model and provide better recommendations to other farmers.
    Your honest responses are valuable for continuous learning!
    """)
    
    # Implementation Status
    st.markdown("### 📋 Did you implement the recommendations?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.feedback_data['took_recommendation'] = st.radio(
            "Have you applied the organic recommendations?",
            options=["Not yet", "Currently implementing", "Fully implemented"],
            index=0 if st.session_state.feedback_data['took_recommendation'] is None else ["Not yet", "Currently implementing", "Fully implemented"].index(st.session_state.feedback_data['took_recommendation']),
            key="rec_implementation"
        )
    
    with col2:
        st.session_state.feedback_data['implementation_timeline'] = st.selectbox(
            "Timeline of implementation:",
            ["Not started", "1-2 weeks ago", "1 month ago", "3 months ago", "6+ months ago", "Full season"],
            index=["Not started", "1-2 weeks ago", "1 month ago", "3 months ago", "6+ months ago", "Full season"].index(st.session_state.feedback_data['implementation_timeline']),
            key="impl_timeline"
        )
    
    st.markdown("---")
    
    # Results & Observations
    st.markdown("### 📊 Results & Observations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Yield Change (%)**")
        st.session_state.feedback_data['yield_change'] = st.slider(
            "What yield change did you observe?",
            -50.0, 100.0,
            st.session_state.feedback_data['yield_change'],
            step=5.0,
            help="-50% = 50% decrease, +50% = 50% increase"
        )
        
        actual_yield = st.session_state.organic_yield * (1 + st.session_state.feedback_data['yield_change']/100)
        st.metric("Actual vs Predicted", f"{actual_yield:.4f} tons/ha", 
             f"Predicted: {st.session_state.organic_yield:.4f} tons/ha")
    
    with col2:
        st.markdown("**Satisfaction Rating**")
        st.session_state.feedback_data['overall_satisfaction'] = st.slider(
            "How satisfied are you with the results?",
            1, 10,
            st.session_state.feedback_data['overall_satisfaction'],
            help="1 = Very unsatisfied, 10 = Extremely satisfied"
        )
        
        satisfaction_emoji = "😞" if st.session_state.feedback_data['overall_satisfaction'] <= 3 else \
                            "😐" if st.session_state.feedback_data['overall_satisfaction'] <= 6 else \
                            "😊" if st.session_state.feedback_data['overall_satisfaction'] <= 8 else "🤩"
        st.write(f"Your rating: {satisfaction_emoji} {st.session_state.feedback_data['overall_satisfaction']}/10")
    
    st.markdown("---")
    
    # Detailed Feedback
    st.markdown("### 💬 Detailed Feedback")
    
    st.session_state.feedback_data['profit_observed'] = st.text_area(
        "🌟 What profits or benefits did you observe?",
        value=st.session_state.feedback_data['profit_observed'],
        placeholder="e.g., Better soil health, increased yield, improved pest resistance, reduced costs, etc.",
        height=100,
        key="profit_text"
    )
    
    st.session_state.feedback_data['drawbacks'] = st.text_area(
        "⚠️ What drawbacks or challenges did you face?",
        value=st.session_state.feedback_data['drawbacks'],
        placeholder="e.g., Higher initial cost, slow results, difficulty sourcing materials, labor intensive, etc.",
        height=100,
        key="drawbacks_text"
    )
    
    st.session_state.feedback_data['recommendations_feedback'] = st.text_area(
        "💡 Suggestions for improving recommendations",
        value=st.session_state.feedback_data['recommendations_feedback'],
        placeholder="e.g., Adjust component ratios, add seasonal variations, consider local climate, etc.",
        height=100,
        key="suggestions_text"
    )
    
    st.markdown("---")
    
    # Final Question
    st.markdown("### 🎯 Overall Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.feedback_data['would_recommend'] = st.radio(
            "Would you recommend this approach to other farmers?",
            options=["Yes, definitely", "Maybe", "No"],
            index=0 if st.session_state.feedback_data['would_recommend'] is None else ["Yes, definitely", "Maybe", "No"].index(st.session_state.feedback_data['would_recommend']),
            key="would_recommend"
        )
    
    with col2:
        if st.session_state.feedback_data['would_recommend'] == "Yes, definitely":
            st.success("🌟 Thank you for your enthusiasm!")
        elif st.session_state.feedback_data['would_recommend'] == "Maybe":
            st.info("📊 Thank you for honest feedback - helps us improve!")
        else:
            st.warning("⚠️ We appreciate you sharing - your insights help us do better!")
    
    # Summary
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Your Feedback Summary</div>', unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        status_icon = "✅" if st.session_state.feedback_data['took_recommendation'] == "Fully implemented" else \
                     "⏳" if st.session_state.feedback_data['took_recommendation'] == "Currently implementing" else "❌"
        st.metric("Implementation", status_icon, st.session_state.feedback_data['took_recommendation'])
    
    with summary_col2:
        st.metric("Actual Yield", f"{actual_yield:.4f} tons/ha", 
                 f"{st.session_state.feedback_data['yield_change']:.0f}%")
    
    with summary_col3:
        st.metric("Satisfaction", f"{st.session_state.feedback_data['overall_satisfaction']}/10", 
                 "rating")
    
    with summary_col4:
        rec_icon = "👍" if st.session_state.feedback_data['would_recommend'] == "Yes, definitely" else \
                   "🤔" if st.session_state.feedback_data['would_recommend'] == "Maybe" else "👎"
        st.metric("Recommend?", rec_icon, st.session_state.feedback_data['would_recommend'])
    
    st.success("✅ Thank you for your valuable feedback! This information helps us improve our AI recommendations for future farmers.")

# ---------------------------------
# NAVIGATION LOGIC
# ---------------------------------
def navigate_to_page(page_num):
    st.session_state.page = page_num

# ---------------------------------
# LANGUAGE & SIDEBAR SETUP
# ---------------------------------
st.sidebar.markdown("## 🌐 Language")
selected_lang = st.sidebar.selectbox(
    "Select Language:",
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index([k for k, v in LANGUAGES.items() if v == st.session_state.language][0] if st.session_state.language in LANGUAGES.values() else 0),
    key="language_selector"
)
st.session_state.language = LANGUAGES[selected_lang]

st.sidebar.markdown("---")

# Sidebar Navigation
st.sidebar.markdown("## 📑 Navigation")
st.sidebar.markdown("---")

page_buttons = {
    1: t("nav_input"),
    2: t("nav_predictions"), 
    3: t("nav_recommendations"),
    4: t("nav_summary"),
    5: t("nav_feedback")
}

for page_num, page_name in page_buttons.items():
    if st.session_state.page == page_num:
        st.sidebar.markdown(f"### **{page_name}** ✓")
    else:
        if st.sidebar.button(page_name, key=f"page_{page_num}", use_container_width=True):
            navigate_to_page(page_num)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 {t('nav_progress')}")
progress = st.session_state.page / 5
st.sidebar.progress(progress, f"Step {st.session_state.page}/5")

# Main app logic
if st.session_state.page == 1:
    page_1_inputs()
elif st.session_state.page == 2:
    page_2_predictions()
elif st.session_state.page == 3:
    page_3_recommendations()
elif st.session_state.page == 4:
    page_4_summary()
elif st.session_state.page == 5:
    page_5_feedback()

# Bottom navigation buttons
st.markdown("---")
st.markdown(f'<div class="page-indicator">📄 {t("page_indicator")} {str(st.session_state.page)} {t("of")} 5</div>', unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.session_state.page > 1:
        if st.button(t("prev_btn"), use_container_width=True, key="prev_btn"):
            navigate_to_page(st.session_state.page - 1)
            st.rerun()

with nav_col4:
    if st.session_state.page < 5:
        if st.button(t("next_btn"), use_container_width=True, key="next_btn"):
            navigate_to_page(st.session_state.page + 1)
            st.rerun()
    else:
        if st.button(t("start_over"), use_container_width=True, key="restart_btn"):
            navigate_to_page(1)
            st.rerun()