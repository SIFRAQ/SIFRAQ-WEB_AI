import streamlit as st
import sys
import subprocess
import os

# ==========================================
# BLOQUE DE INSTALACIÓN DE EMERGENCIA (CORREGIDO)
# ==========================================
try:
    import detectron2
except ImportError:
    st.warning("⚠️ Configurando entorno de Inteligencia Artificial... (Esto tardará unos minutos, por favor espera)")
    
    # 1. Forzamos modo CPU para ahorrar memoria durante la instalación
    os.environ["FORCE_CUDA"] = "0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""
    
    # 2. Instalamos usando "--no-build-isolation" para que detecte el PyTorch que ya tenemos
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/facebookresearch/detectron2.git@v0.6",
            "--no-build-isolation", 
            "--no-cache-dir"
        ])
        # 3. Recargamos la app
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Error crítico instalando IA: {e}")
        st.stop()

# ---------------------------------------------
# IMPORTACIONES NORMALES
# ---------------------------------------------
import cv2
import numpy as np
import pandas as pd
import torch
import time
import io
import qrcode
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, KeepTogether, PageBreak
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus.flowables import Image as ReportLabImage
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

# --- CONFIGURACIÓN DE DETECTRON2 ---
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ... (EL RESTO DE TU CÓDIGO SIGUE IGUAL) ...

#---------------------------------------------------------------------
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import os
import time
import io
import qrcode
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
# Importamos PageBreak para forzar el salto de página
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, KeepTogether, PageBreak
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus.flowables import Image as ReportLabImage
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

# --- CONFIGURACIÓN DE DETECTRON2 ---
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(page_title="SIFRAQ - Minería Inteligente", layout="wide", page_icon="🪨")

# Estilos CSS Profesionales (Mining Tech Theme - High Contrast for Dark Mode)
st.markdown("""
<style>
    /* Fondo General */
    .stApp {
        background-color: #0E1117;
        color: #F0F2F6; /* Blanco humo para texto general */
    }

    /* Títulos Principales (H1, H2, H3) - Colores Cyan y Azul Eléctrico */
    h1, h2, h3 {
        color: #00E5FF !important; /* Cyan Brillante */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    
    /* Subtítulos o H4, H5 */
    h4, h5, h6 {
        color: #2979FF !important; /* Azul Eléctrico */
        font-weight: 600;
    }

    /* Textos de Párrafos y Span */
    p, div, span {
        color: #E0E0E0; /* Gris muy claro */
    }

    /* Etiquetas de Inputs (Text Input, Number Input, etc) */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stCheckbox > label, .stFileUploader > label {
        color: #00E5FF !important; /* Cyan para que resalte sobre el negro */
        font-weight: bold;
    }

    /* Métricas (st.metric) */
    [data-testid="stMetricLabel"] {
        color: #2979FF !important; /* Etiqueta azul */
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* Valor en blanco puro */
    }

    /* Botones Estilizados */
    .stButton>button {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%); /* Gradiente Cyan-Azul */
        color: white;
        border: 1px solid #00E5FF;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00E5FF 0%, #004d99 100%);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        border-color: #FFFFFF;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Indicador de pasos en Sidebar */
    .step-indicator {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background: #21262D;
        color: #8B949E;
        font-size: 0.9em;
    }
    .step-indicator.active {
        background: rgba(0, 229, 255, 0.1);
        border-left: 4px solid #00E5FF;
        color: #FFFFFF;
        font-weight: bold;
    }

    /* Modal de Bienvenida */
    .welcome-modal {
        padding: 20px;
        background-color: #1F2937;
        border-radius: 10px;
        border: 1px solid #00E5FF;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Eliminar ejes blancos de Plotly para integración oscura */
    .js-plotly-plot .plotly .main-svg { 
        background: transparent !important; 
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GESTIÓN DE ESTADO
# ==========================================
if 'step' not in st.session_state: st.session_state.step = 0
if 'user_info' not in st.session_state: st.session_state.user_info = None
if 'project_data' not in st.session_state: st.session_state.project_data = {}
if 'image_original' not in st.session_state: st.session_state.image_original = None
if 'image_preprocessed' not in st.session_state: st.session_state.image_preprocessed = None
if 'px_per_cm' not in st.session_state: st.session_state.px_per_cm = None
if 'scale_line_coords' not in st.session_state: st.session_state.scale_line_coords = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'conclusiones_texto' not in st.session_state: st.session_state.conclusiones_texto = ""
if 'esponjamiento_calc' not in st.session_state: st.session_state.esponjamiento_calc = 0
if 'welcome_shown' not in st.session_state: st.session_state.welcome_shown = False

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def reset_app():
    user = st.session_state.user_info
    for key in list(st.session_state.keys()):
        if key not in ['user_info']:
            del st.session_state[key]
    st.session_state.step = 1
    st.session_state.user_info = user
    st.rerun()

# ==========================================
# LÓGICA CIENTÍFICA MEJORADA
# ==========================================

class FragmentAnalyzerEnhanced:
    """Analizador mejorado basado en el código Colab proporcionado"""
    
    def __init__(self, scale_diameter_cm=10.0, size_threshold_cm=2.0):
        self.scale_diameter_cm = scale_diameter_cm
        self.size_threshold_cm = size_threshold_cm
        self.pixels_per_cm = None
        self.scale_detected = False

    def pixels_to_cm(self, measurement_pixels):
        """Convertir medición en píxeles a centímetros"""
        if self.pixels_per_cm is None:
            return None
        return measurement_pixels / self.pixels_per_cm

    def calculate_fragment_size(self, fragment_mask):
        """Calcular tamaño real del fragmento en cm"""
        try:
            area_pixels = np.sum(fragment_mask)
            equivalent_diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
            equivalent_diameter_cm = self.pixels_to_cm(equivalent_diameter_pixels)
            
            if equivalent_diameter_cm > self.size_threshold_cm:
                size_category = "Sobredimensionado"
            else:
                size_category = "Óptimo"
                
            return {
                'area_pixels': area_pixels,
                'equivalent_diameter_pixels': equivalent_diameter_pixels,
                'equivalent_diameter_cm': equivalent_diameter_cm,
                'size_category': size_category
            }
        except Exception as e:
            print(f"Error cálculo tamaño: {e}")
            return None

def preprocesar_imagen(image_np):
    """Preprocesamiento mejorado basado en Colab"""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # CLAHE para mejora de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr, enhanced

@st.cache_resource
def cargar_modelo():
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        
        current_dir = os.getcwd()
        model_path = os.path.join(current_dir, "model", "model_final.pth")
        
        if not os.path.exists(model_path):
            st.error("No se encuentra model_final.pth en la carpeta model/")
            return None
            
        cfg.MODEL.WEIGHTS = model_path
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000 
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        return DefaultPredictor(cfg)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

def calcular_esponjamiento(densidad_insitu, rqd, familias):
    """Calcula densidad suelta y esponjamiento"""
    try:
        factor_rqd = 1 - (rqd / 100) * 0.15
        factor_familias = 1 - (familias * 0.05)
        factor_esponjamiento = factor_rqd * factor_familias
        densidad_suelta = densidad_insitu * (0.55 + factor_esponjamiento * 0.25)
        esponjamiento = ((densidad_insitu / densidad_suelta) - 1) * 100
        return densidad_suelta, esponjamiento
    except:
        return 0, 0

def generar_estadisticas_avanzadas(df, tamano_optimo):
    """Genera estadísticas avanzadas basadas en el análisis de fragmentos"""
    if df.empty:
        return {}
    
    try:
        df_copy = df.copy()
        df_copy["Tamaño (cm)"] = pd.to_numeric(df_copy["Tamaño (cm)"], errors='coerce')
        df_clean = df_copy.dropna(subset=["Tamaño (cm)"])
        
        if df_clean.empty:
            return {}
        
        diametros = df_clean["Tamaño (cm)"].values
        optimos = df_clean[df_clean["Categoría"] == "Óptimo"]
        sobredim = df_clean[df_clean["Categoría"] == "Sobredimensionado"]
        
        # Calcular percentiles
        if len(diametros) > 0:
            percentiles = np.percentile(diametros, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        else:
            percentiles = np.zeros(10)
        
        estadisticas = {
            'total_fragments': len(df_clean),
            'fragments_with_size': len(diametros),
            'optimos': len(optimos),
            'sobredimensionados': len(sobredim),
            'porcentaje_optimos': (len(optimos) / len(df_clean)) * 100 if len(df_clean) > 0 else 0,
            'porcentaje_sobredim': (len(sobredim) / len(df_clean)) * 100 if len(df_clean) > 0 else 0,
            'mean_diameter_cm': float(np.mean(diametros)) if len(diametros) > 0 else 0.0,
            'median_diameter_cm': float(np.median(diametros)) if len(diametros) > 0 else 0.0,
            'std_diameter_cm': float(np.std(diametros)) if len(diametros) > 0 else 0.0,
            'min_diameter_cm': float(np.min(diametros)) if len(diametros) > 0 else 0.0,
            'max_diameter_cm': float(np.max(diametros)) if len(diametros) > 0 else 0.0,
            'd50': float(percentiles[4]),
            'd80': float(percentiles[7]),
            'uniformidad': float(percentiles[7] / percentiles[0]) if percentiles[0] > 0 else 0.0,
            'percentiles': [float(p) for p in percentiles],
            'diametros_array': [float(d) for d in diametros]
        }
        
        return estadisticas
        
    except Exception as e:
        return {}

def debug_data():
    """Función para debug de datos"""
    if st.session_state.analysis_results:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**DEBUG INFO:**")
        res = st.session_state.analysis_results
        st.sidebar.write(f"DataFrame shape: {res['df'].shape}")

# ==========================================
# UI COMPONENTS
# ==========================================

def sidebar_menu():
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_column_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("SISTEMA DE NAVEGACIÓN")
    
    steps = ["Login", "Proyecto", "Carga", "Calibración", "Parámetros", "Resultados", "Reporte"]
    
    for i, step_name in enumerate(steps):
        status = "active" if st.session_state.step == i else ""
        icon = "🟢" if st.session_state.step > i else "⚪" if st.session_state.step < i else "🔵"
        st.sidebar.markdown(f'<div class="step-indicator {status}">{icon} {step_name}</div>', unsafe_allow_html=True)

    if st.session_state.user_info:
        st.sidebar.markdown("---")
        st.sidebar.write(f"👤 **Usuario:** {st.session_state.user_info}")

# ==========================================
# SLIDES
# ==========================================

def show_landing():
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("# Fragmentación analizada con precisión inteligente")
        st.markdown("""
        <div style='text-align: justify; font-size: 1.15em; color: #B0B0B0; line-height: 1.6;'>
        Caracteriza, mide y optimiza la distribución granulométrica con tecnología de inteligencia artificial, 
        análisis de datos y automatización en <b>SIFRAQ</b>. 
        Toma decisiones basadas en datos precisos para  
        mejorar la productividad, eficiencia y seguridad en tus operaciones mineras. 
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🔵 Iniciar Análisis de Fragmentación", use_container_width=True):
            with st.spinner("Conectando IA..."):
                time.sleep(2)
                st.session_state.user_info = "Ingeniero de Minas"
                st.session_state.step = 1
                st.rerun()

    with col2:
        video_path = os.path.join("assets", "Video_portada.mp4")
        logo_path = os.path.join("assets", "logo.png")
        
        if os.path.exists(video_path):
            st.video(video_path, autoplay=True, loop=True, muted=True)
        elif os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)

def show_project_data():
    if not st.session_state.welcome_shown:
        st.markdown(f"""
        <div class="welcome-modal">
            <h3>👋 ¡Bienvenido a SIFRAQ, {st.session_state.user_info}!</h3>
            <p>Listo para optimizar tu voladura hoy.</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True
        time.sleep(1.5)

    st.markdown("## 📂 Datos del Proyecto")
    col1, col2 = st.columns(2)
    with col1:
        nombre = st.text_input("Nombre del Proyecto / Mina", value=st.session_state.project_data.get("nombre", ""))
    with col2:
        ubicacion = st.text_input("Ubicación / Frente", value=st.session_state.project_data.get("ubicacion", ""))
    
    st.markdown("---")
    if st.button("Guardar y Continuar ➡️"):
        if nombre and ubicacion:
            st.session_state.project_data = {"nombre": nombre, "ubicacion": ubicacion}
            next_step()
            st.rerun()
        else:
            st.warning("Completa los campos para continuar.")

def show_upload():
    st.markdown("## 📸 Carga de Imagen")
    uploaded_file = st.file_uploader("Sube la imagen del material volado (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)
        st.session_state.image_original = image_np
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Imagen Original", use_column_width=True)
        with col2:
            img_bgr, img_gray = preprocesar_imagen(image_np)
            st.session_state.image_preprocessed = img_bgr
            st.image(img_gray, caption="Preprocesamiento IA (Contraste Mejorado)", use_column_width=True)
            
        st.markdown("---")
        c1, c2 = st.columns([1, 5])
        c1.button("⬅️ Atrás", on_click=prev_step)
        c2.button("Continuar a Calibración ➡️", on_click=next_step)

def show_calibration():
    st.markdown("## 📏 Calibración de Precisión")
    st.info("Dibuja una **LÍNEA RECTA (Verde)** sobre el objeto de referencia.")
    
    if st.session_state.image_original is not None:
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=Image.fromarray(st.session_state.image_original),
            update_streamlit=True,
            height=600,
            drawing_mode="line",
            key="canvas_calib",
        )
    else:
        st.error("Primero carga una imagen en el paso anterior.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        medida_real = st.number_input("Longitud real del objeto de referencia:", min_value=0.1, value=10.0, step=0.1)
    with col2:
        unidad = st.selectbox("Unidad:", ["cm", "mm", "m"])
    
    calibrado = False
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        
        if 'x1' in obj and 'y1' in obj and 'x2' in obj and 'y2' in obj:
            x1, y1, x2, y2 = obj['x1'], obj['y1'], obj['x2'], obj['y2']
        else:
            x1, y1 = obj['left'], obj['top']
            x2, y2 = x1 + obj['width'], y1 + obj['height']
        
        dist_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        factor = 1.0
        if unidad == "mm": factor = 0.1
        if unidad == "m": factor = 100.0
        
        px_cm = dist_px / (medida_real * factor)
        st.session_state.px_per_cm = px_cm
        
        st.session_state.scale_line_coords = {
            'x1': int(x1), 'y1': int(y1), 
            'x2': int(x2), 'y2': int(y2),
            'color': (0, 255, 0) 
        }
        
        st.success(f"✅ Calibrado: {px_cm:.2f} px/cm | Longitud referencia: {medida_real} {unidad}")
        calibrado = True

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    c1.button("⬅️ Atrás", on_click=prev_step)
    c2.button("Definir Parámetros ➡️", on_click=next_step, disabled=not calibrado)

def show_parameters():
    st.markdown("## ⚙️ Parámetros de Análisis")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Configuración de Fragmentación")
        tamano_optimo = st.number_input("Tamaño Óptimo Máximo (cm)", value=2.0, min_value=0.1, step=0.1)
        
    with c2:
        st.subheader("Parámetros Geomecánicos")
        usar_geo = st.checkbox("Incluir análisis de esponjamiento", value=True)
        densidad, rqd, familias = 2.7, 70, 3
        
        if usar_geo:
            st.markdown("[Consultar Tabla de Densidades](https://www.geovirtual2.cl/EXPLORAC/TEXT/gravi-tabla-10.png)") 
            densidad = st.number_input("Densidad In-Situ (g/cm³)", value=2.7, min_value=1.0, max_value=5.0, step=0.1)
            rqd = st.number_input("RQD (%)", 0, 100, 70)
            familias = st.number_input("N° Familias de Discontinuidades", 1, 10, 3)

    st.markdown("---")
    if st.button("🚀 EJECUTAR ANÁLISIS COMPLETO", use_container_width=True, type="primary"):
        with st.spinner("SIFRAQ AI procesando imagen y analizando fragmentación..."):
            try:
                analyzer = FragmentAnalyzerEnhanced(size_threshold_cm=tamano_optimo)
                
                if st.session_state.px_per_cm:
                    analyzer.pixels_per_cm = st.session_state.px_per_cm
                    analyzer.scale_detected = True
                
                predictor = cargar_modelo()
                if not predictor:
                    st.error("No se pudo cargar el modelo de IA")
                    return
                    
                predictor.model.roi_heads.score_thresh_test = 0.5
                outputs = predictor(st.session_state.image_preprocessed)
                
                masks = outputs["instances"].pred_masks.to("cpu").numpy()
                scores = outputs["instances"].scores.to("cpu").numpy()
                classes = outputs["instances"].pred_classes.to("cpu").numpy()
                
                img_res = st.session_state.image_original.copy()
                data = []
                
                COLOR_OPTIMO = (255, 0, 0)      # AZUL en BGR
                COLOR_SOBREDIM = (0, 0, 255)    # ROJO en BGR  
                COLOR_ESCALA = (0, 255, 0)      # VERDE en BGR
                
                fragment_count = 0
                
                for i, (mask, score, class_id) in enumerate(zip(masks, scores, classes)):
                    if class_id == 1:  # Fragmento
                        size_data = analyzer.calculate_fragment_size(mask)
                        
                        if size_data and size_data['equivalent_diameter_cm'] is not None:
                            diam_cm = size_data['equivalent_diameter_cm']
                            categoria = size_data['size_category']
                            
                            if categoria == "Óptimo":
                                color = COLOR_OPTIMO
                            else:
                                color = COLOR_SOBREDIM
                            
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(img_res, contours, -1, color, 2)
                            
                            ys, xs = np.where(mask)
                            if len(ys) > 0 and len(xs) > 0:
                                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                                
                                data.append({
                                    "ID": fragment_count, 
                                    "Tamaño (cm)": round(diam_cm, 2), 
                                    "Categoría": categoria, 
                                    "cx": cx, "cy": cy, 
                                    "Area_px": size_data['area_pixels'], 
                                    "Confianza": round(score, 3)
                                })
                                fragment_count += 1
                
                # DIBUJAR LÍNEA DE ESCALA PERO NO LEYENDA DENTRO DE IMAGEN
                if st.session_state.scale_line_coords:
                    scale_coords = st.session_state.scale_line_coords
                    x1, y1, x2, y2 = scale_coords['x1'], scale_coords['y1'], scale_coords['x2'], scale_coords['y2']
                    cv2.line(img_res, (x1, y1), (x2, y2), COLOR_ESCALA, 4)
                    cv2.putText(img_res, "ESCALA", (x1, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ESCALA, 2)
                
                df = pd.DataFrame(data)
                
                esponjamiento = 0
                if usar_geo:
                    _, esponjamiento = calcular_esponjamiento(densidad, rqd, familias)
                    st.session_state.esponjamiento_calc = esponjamiento
                
                if not df.empty:
                    pct_optimos = (len(df[df["Categoría"]=="Óptimo"]) / len(df)) * 100
                    if pct_optimos >= 80:
                        conclusion = "✅ EXCELENTE fragmentación - Parámetros de voladura óptimos"
                    elif pct_optimos >= 60:
                        conclusion = "⚠️ BUENA fragmentación - Considerar ajustes menores"
                    else:
                        conclusion = "❌ FRAGMENTACIÓN DEFICIENTE - Revisar parámetros de voladura"
                        
                    conclusion += f" | {pct_optimos:.1f}% fragmentos óptimos"
                    st.session_state.conclusiones_texto = conclusion
                else:
                    st.session_state.conclusiones_texto = "⚠️ No se detectaron fragmentos - Revisar parámetros de detección"
                
                estadisticas = generar_estadisticas_avanzadas(df, tamano_optimo)

                st.session_state.analysis_results = {
                    "df": df, 
                    "img": img_res, 
                    "params": {
                        "densidad": densidad, "rqd": rqd, "familias": familias, 
                        "optimo": tamano_optimo, "confianza": 50
                    },
                    "estadisticas": estadisticas
                }
                
                next_step()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error en el análisis: {str(e)}")
                st.info("Por favor, verifica que la imagen sea adecuada y reintenta.")

def show_results():
    st.markdown("## 📊 Resultados e Interpretación")
    
    res = st.session_state.analysis_results
    if not res or res["df"].empty:
        st.error("No se detectaron fragmentos en el análisis. Ajusta los parámetros y reintenta.")
        st.button("⬅️ Ajustar Parámetros", on_click=prev_step)
        return
        
    df = res["df"]
    stats = res["estadisticas"]
    params = res["params"]
    
    debug_data()
    
    # ----------------------------------------------------
    # PESTAÑAS SOLICITADAS
    # ----------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🪨 Fragmentos Detectados", 
        "📈 Análisis Estadístico", 
        "📉 Curvas Granulométricas",
        "⛏️ Análisis Geomecánico"
    ])
    
    with tab1:
        st.markdown("### Visualización y Detalles")
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("Total Fragmentos", f"{stats['total_fragments']}")
        with col2: 
            st.metric("Eficiencia Óptima", f"{stats['porcentaje_optimos']:.1f}%")
        with col3: 
            st.metric("Sobredimensionados", f"{stats['sobredimensionados']} ({stats['porcentaje_sobredim']:.1f}%)")
        with col4: 
            area_promedio = np.mean(df['Area_px'] / (st.session_state.px_per_cm**2)) if st.session_state.px_per_cm else 0
            st.metric("Área Promedio (cm²)", f"{area_promedio:.2f}")

        with st.expander("👁️ Visualización de Imagen Analizada", expanded=True):
            st.markdown("""
            **Leyenda:** 🟢 <span style='color:green'>Escala</span> | 🔵 <span style='color:blue'>Óptimos (≤ {:.1f} cm)</span> | 🔴 <span style='color:red'>Sobredimensionados (> {:.1f} cm)</span>
            """.format(params["optimo"], params["optimo"]), unsafe_allow_html=True)
            
            img_rgb = cv2.cvtColor(res["img"], cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Fragmentación Detectada", use_column_width=True)

        st.markdown("#### Tabla de Detalles")
        st.dataframe(df[["ID", "Tamaño (cm)", "Categoría", "Confianza"]], use_container_width=True)

    with tab2:
        st.markdown("### 📈 Análisis Estadístico")
        if df.empty:
            st.warning("No hay datos para mostrar análisis estadístico")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🥧 Distribución por Categoría")
                try:
                    optimos_count = len(df[df["Categoría"] == "Óptimo"])
                    sobredim_count = len(df[df["Categoría"] == "Sobredimensionado"])
                    if optimos_count + sobredim_count > 0:
                        fig_pie = go.Figure(data=[
                            go.Pie(
                                labels=['Óptimo', 'Sobredimensionado'],
                                values=[optimos_count, sobredim_count],
                                hole=0.4,
                                marker=dict(colors=['blue', 'red'])
                            )
                        ])
                        fig_pie.update_layout(title="Distribución por Categoría", height=400, showlegend=True)
                        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart_forced")
                    else:
                        st.warning("No hay datos para el gráfico de pie")
                except Exception as e:
                    st.error(f"Error en gráfico de pie: {e}")
            
            with col2:
                st.markdown("#### 📊 Histograma de Tamaños")
                try:
                    tamaños_optimos = df[df["Categoría"] == "Óptimo"]["Tamaño (cm)"].tolist()
                    tamaños_sobredim = df[df["Categoría"] == "Sobredimensionado"]["Tamaño (cm)"].tolist()
                    fig_hist = go.Figure()
                    if tamaños_optimos:
                        fig_hist.add_trace(go.Histogram(x=tamaños_optimos, name='Óptimo', marker_color='blue', opacity=0.7, nbinsx=20))
                    if tamaños_sobredim:
                        fig_hist.add_trace(go.Histogram(x=tamaños_sobredim, name='Sobredimensionado', marker_color='red', opacity=0.7, nbinsx=20))
                    
                    fig_hist.update_layout(
                        title="Distribución de Tamaños de Fragmentos",
                        xaxis_title="Tamaño (cm)",
                        yaxis_title="Frecuencia",
                        barmode='overlay',
                        height=400,
                        bargap=0.1
                    )
                    config = {'displayModeBar': True, 'responsive': True}
                    st.plotly_chart(fig_hist, use_container_width=True, config=config, key="histogram_final")
                except Exception as e:
                    st.error(f"Error en histograma: {e}")
            
            st.markdown("---")
            st.markdown("#### 📋 Métricas Estadísticas Detalladas")
            if stats and isinstance(stats, dict):
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Diámetro Promedio", f"{stats.get('mean_diameter_cm', 0):.2f} cm")
                    st.metric("Diámetro Mínimo", f"{stats.get('min_diameter_cm', 0):.2f} cm")
                with col_stat2:
                    st.metric("Diámetro Mediano", f"{stats.get('median_diameter_cm', 0):.2f} cm")
                    st.metric("Diámetro Máximo", f"{stats.get('max_diameter_cm', 0):.2f} cm")
                with col_stat3:
                    st.metric("Desviación Estándar", f"{stats.get('std_diameter_cm', 0):.2f} cm")
                    coef_var = (stats.get('std_diameter_cm', 0)/stats.get('mean_diameter_cm', 1))*100 if stats.get('mean_diameter_cm', 0) > 0 else 0
                    st.metric("Coef. Variación", f"{coef_var:.1f}%")

    with tab3:
        st.markdown("### 📉 Curva Granulométrica")
        if df.empty or "Tamaño (cm)" not in df.columns:
            st.warning("No hay datos para generar curva granulométrica")
        else:
            try:
                tamaños_lista = df["Tamaño (cm)"].tolist()
                if not tamaños_lista:
                    st.warning("No hay valores para la curva")
                else:
                    sorted_sizes = sorted(tamaños_lista)
                    cumulative_passing = []
                    for i in range(len(sorted_sizes)):
                        percentage = (i + 1) / len(sorted_sizes) * 100
                        cumulative_passing.append(percentage)
                    
                    fig_curve = go.Figure()
                    fig_curve.add_trace(go.Scatter(
                        x=sorted_sizes, y=cumulative_passing,
                        mode='lines+markers', line=dict(color='#4DA6FF', width=3),
                        marker=dict(size=6, color='#4DA6FF'), name='% Pasante Acumulado'
                    ))
                    
                    d50 = np.percentile(sorted_sizes, 50)
                    d80 = np.percentile(sorted_sizes, 80)
                    
                    fig_curve.add_vline(x=d50, line_dash="dash", line_color="orange", annotation_text=f"D50 = {d50:.2f} cm", annotation_position="top right")
                    fig_curve.add_vline(x=d80, line_dash="dash", line_color="green", annotation_text=f"D80 = {d80:.2f} cm", annotation_position="top right")
                    
                    fig_curve.update_layout(
                        title="Curva Granulométrica - Pasante Acumulado",
                        xaxis_title="Tamaño de Partícula (cm)",
                        yaxis_title="% Pasante Acumulado",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': True, 'responsive': True}, key="curve_final")
                    
                    st.markdown("#### 📏 Parámetros Granulométricos")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("D50 (Tamaño Medio)", f"{d50:.2f} cm")
                    col2.metric("D80 (Tamaño Control)", f"{d80:.2f} cm")
                    d10 = np.percentile(sorted_sizes, 10)
                    uniformidad = d80 / d10 if d10 > 0 else 0
                    col3.metric("Coeficiente Uniformidad", f"{uniformidad:.2f}")
            except Exception as e:
                st.error(f"Error generando curva: {e}")

    with tab4:
        st.markdown("### Evaluación Técnica")
        if st.session_state.esponjamiento_calc > 0:
            c_geo1, c_geo2 = st.columns(2)
            with c_geo1:
                st.info(f"""
                **Parámetros de Entrada:**
                - Densidad In-Situ: {params['densidad']} g/cm³
                - RQD: {params['rqd']}%
                - Familias: {params['familias']}
                """)
            with c_geo2:
                densidad_suelta = params["densidad"] / (1 + st.session_state.esponjamiento_calc/100)
                st.success(f"""
                **Resultados Calculados:**
                - Esponjamiento: **{st.session_state.esponjamiento_calc:.2f}%**
                - Densidad Suelta: **{densidad_suelta:.2f} g/cm³**
                """)
            
            st.markdown("#### Conclusiones y Recomendaciones")
            st.info(st.session_state.conclusiones_texto)
            
            if stats['porcentaje_sobredim'] > 15:
                st.error("⚠️ **ALERTA:** El porcentaje de sobredimensionado supera el 15%.")
                st.markdown("""
                **Acciones Sugeridas:**
                1. Revisar malla de perforación (Burden/Espaciamiento).
                2. Verificar tiempos de retardo en la secuencia de encendido.
                3. Considerar uso de tacos de aire o mejor confinamiento.
                """)
            else:
                st.success("✅ **ESTADO:** La fragmentación está dentro de rangos operativos aceptables.")
        else:
            st.warning("No se activó el cálculo geomecánico en los parámetros.")

    st.markdown("---")
    c1, c2 = st.columns([1, 5])
    c1.button("⬅️ Reanalizar", on_click=prev_step)
    c2.button("Generar Reporte Final ➡️", on_click=next_step)

# ==========================================
# GENERACIÓN DE REPORTE PDF PROFESIONAL
# ==========================================

def _save_matplotlib_fig_to_temp(fig, suffix=".png"):
    """Guarda figura matplotlib en archivo temporal"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fig.tight_layout()
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return tmp.name

def draw_background(canvas, doc):
    """Dibuja el fondo, marca de agua (LOGO GRANDE) y logo de esquina"""
    canvas.saveState()
    
    # 1. LOGO EN LA ESQUINA SUPERIOR DERECHA (Pequeño)
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        canvas.drawImage(logo_path, A4[0] - 4*cm, A4[1] - 3*cm, width=2.5*cm, height=2.5*cm, mask='auto', preserveAspectRatio=True)

    # 2. MARCA DE AGUA (LOGO GRANDE CENTRADO)
    if os.path.exists(logo_path):
        canvas.setFillAlpha(0.15)  # AUMENTADO DE 0.08 A 0.15 PARA MEJOR VISIBILIDAD SIN RUIDO
        # Tamaño grande para el fondo
        wm_width = 15*cm
        wm_height = 15*cm
        # Centrado en la página A4
        x_centered = (A4[0] - wm_width) / 2
        y_centered = (A4[1] - wm_height) / 2
        canvas.drawImage(logo_path, x_centered, y_centered, width=wm_width, height=wm_height, mask='auto', preserveAspectRatio=True)
    else:
        # Fallback de texto si no hay logo
        canvas.translate(A4[0]/2, A4[1]/2)
        canvas.rotate(45)
        canvas.setFont("Helvetica-Bold", 60)
        canvas.setFillColor(colors.lightgrey, alpha=0.1)
        canvas.drawCentredString(0, 0, "SIFRAQ")
    
    canvas.restoreState()

def crear_pdf_profesional():
    """Genera PDF formal con comparativa de imágenes, gráficos y marca de agua."""
    buffer = io.BytesIO()
    
    # Configuración de márgenes
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4, 
        rightMargin=2*cm, leftMargin=2*cm, 
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )

    # --- ESTILOS FORMALES UNIFICADOS ---
    styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        'CustomTitle', parent=styles['Title'], fontName='Helvetica-Bold', 
        fontSize=18, textColor=HexColor("#003366"), alignment=TA_CENTER, spaceAfter=12
    )
    
    style_subtitle = ParagraphStyle(
        'CustomSubtitle', parent=styles['Normal'], fontName='Helvetica',
        fontSize=12, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=20
    )

    style_h2 = ParagraphStyle(
        'CustomH2', parent=styles['Heading2'], fontName='Helvetica-Bold', 
        fontSize=12, textColor=HexColor("#003366"), spaceBefore=12, spaceAfter=6,
        borderPadding=0, borderColor=HexColor("#003366"), borderWidth=0
    )

    # Estilo nuevo para títulos de gráficos (H3)
    style_h3 = ParagraphStyle(
        'CustomH3', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=10, textColor=colors.black, spaceBefore=10, spaceAfter=4, alignment=TA_LEFT
    )
    
    style_normal = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'], fontName='Helvetica', 
        fontSize=10, leading=14, alignment=TA_JUSTIFY
    )
    
    # ESTILO DE VIÑETA PROFESIONAL (Bullet Point)
    style_bullet_pro = ParagraphStyle(
        'BulletPro',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        leftIndent=20,      # Sangría del texto
        firstLineIndent=0,
        bulletIndent=10,    # Sangría de la viñeta
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    style_bold = ParagraphStyle(
        'CustomBold', parent=style_normal, fontName='Helvetica-Bold'
    )

    story = []

    # 1. ENCABEZADO
    story.append(Paragraph("INFORME TÉCNICO DE FRAGMENTACIÓN", style_title))
    story.append(Paragraph("SISTEMA DE PREDICCIÓN DE FRAGMENTOS POST VOLADURA", style_subtitle))
    story.append(Spacer(1, 10))

    # 2. INFORMACIÓN DEL PROYECTO
    fecha_analisis = datetime.now().strftime('%d/%m/%Y %H:%M')
    info_data = [
        [Paragraph(f"<b>Proyecto:</b> {st.session_state.project_data.get('nombre', 'N/A')}", style_normal),
         Paragraph(f"<b>Fecha:</b> {fecha_analisis}", style_normal)],
        [Paragraph(f"<b>Ubicación:</b> {st.session_state.project_data.get('ubicacion', 'N/A')}", style_normal),
         Paragraph(f"<b>Usuario:</b> {st.session_state.user_info}", style_normal)]
    ]
    t_info = Table(info_data, colWidths=[10*cm, 7*cm])
    t_info.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 15))
    story.append(Paragraph("_" * 78, style_normal))
    story.append(Spacer(1, 15))

    # 3. ANÁLISIS VISUAL COMPARATIVO
    # Bloque KeepTogether para que título e imagen no se separen
    content_visual = []
    content_visual.append(Paragraph("1. ANÁLISIS VISUAL", style_h2))
    
    img_orig_path = None
    img_res_path = None
    
    if st.session_state.image_original is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
            im_pil = Image.fromarray(st.session_state.image_original)
            im_pil.save(tmp1.name, quality=85)
            img_orig_path = tmp1.name

    res = st.session_state.analysis_results
    if res and 'img' in res:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            im_pil_res = Image.fromarray(cv2.cvtColor(res["img"], cv2.COLOR_BGR2RGB))
            im_pil_res.save(tmp2.name, quality=85)
            img_res_path = tmp2.name

    if img_orig_path and img_res_path:
        img_width = 8*cm
        img_height = 6*cm
        imgs_data = [
            [ReportLabImage(img_orig_path, width=img_width, height=img_height),
             ReportLabImage(img_res_path, width=img_width, height=img_height)],
            [Paragraph("<b>Imagen Original</b>", ParagraphStyle('center', parent=style_normal, alignment=TA_CENTER)),
             Paragraph("<b>Análisis + Escala</b>", ParagraphStyle('center', parent=style_normal, alignment=TA_CENTER))]
        ]
        t_imgs = Table(imgs_data, colWidths=[8.5*cm, 8.5*cm])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        content_visual.append(t_imgs)
        
        # --- LEYENDA MODIFICADA (Solo Azul y Rojo) ---
        content_visual.append(Spacer(1, 8))
        optimo_val_legend = res['params']['optimo'] if res else 0.0
        
        # Columnas: [Azul] Texto | [Rojo] Texto
        legend_data = [
            ['', f'Óptimo (≤ {optimo_val_legend} cm)', 
             '', f'Sobredimensionado (> {optimo_val_legend} cm)']
        ]
        
        t_legend = Table(legend_data, colWidths=[0.6*cm, 4*cm, 0.6*cm, 5*cm])
        t_legend.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.darkgrey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            # Azul
            ('BACKGROUND', (0,0), (0,0), colors.blue),
            # Rojo
            ('BACKGROUND', (2,0), (2,0), colors.red),
            ('BOX', (0,0), (0,0), 0.5, colors.black),
            ('BOX', (2,0), (2,0), 0.5, colors.black),
        ]))
        # Centrar la leyenda
        t_legend.hAlign = 'CENTER'
        content_visual.append(t_legend)
        # ---------------------------------------------
    else:
        content_visual.append(Paragraph("No hay imágenes disponibles para comparar.", style_normal))
    
    content_visual.append(Spacer(1, 12))
    story.append(KeepTogether(content_visual))


    # 4. TABLA DE DATOS ESTADÍSTICOS
    content_stats = []
    content_stats.append(Paragraph("2. RESUMEN ESTADÍSTICO", style_h2))
    
    stats = res.get("estadisticas", {}) if res else {}
    optimo_ref = res['params']['optimo'] if res else 0.0
    
    table_data = [
        ['PARÁMETRO', 'VALOR CALCULADO', 'UNIDAD'],
        ['Total Fragmentos Detectados', str(stats.get('total_fragments', 0)), 'Unidades'],
        ['Tamaño Óptimo (≤ {:.1f} cm)'.format(optimo_ref), 
         f"{stats.get('optimos', 0)} ({stats.get('porcentaje_optimos',0):.1f}%)", '%'],
        ['Sobredimensionado (> {:.1f} cm)'.format(optimo_ref), 
         f"{stats.get('sobredimensionados', 0)} ({stats.get('porcentaje_sobredim',0):.1f}%)", '%'],
        ['D50 (Tamaño Medio)', f"{stats.get('d50', 0.0):.2f}", 'cm'],
        ['D80 (Tamaño de Control)', f"{stats.get('d80', 0.0):.2f}", 'cm'],
        ['Coeficiente de Uniformidad', f"{stats.get('uniformidad', 0.0):.2f}", '-']
    ]
    
    t_stats = Table(table_data, colWidths=[8*cm, 5*cm, 4*cm])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#0E1117")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    content_stats.append(t_stats)
    content_stats.append(Spacer(1, 12))
    story.append(KeepTogether(content_stats))

    # --- FORZAR SALTO DE PÁGINA ANTES DE LA SECCIÓN 3 ---
    story.append(PageBreak()) 
    # ---------------------------------------------------

    # 5. GRÁFICOS INSERTADOS (Separados y Rotulados)
    diametros = stats.get('diametros_array', [])
    sorted_sizes = sorted(diametros) if diametros else []
    
    # Header Principal Sección 3
    story.append(Paragraph("3. GRÁFICOS DE COMPORTAMIENTO", style_h2))
    
    # -- 3.1 Diagrama de Distribución --
    content_g1 = []
    content_g1.append(Paragraph("3.1. Distribución Porcentual de Material", style_h3))
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    labels = ['Óptimo', 'Sobredimensionado']
    values = [stats.get('optimos', 0), stats.get('sobredimensionados', 0)]
    if sum(values) > 0:
        ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4DA6FF', '#FF4D4D'], startangle=90)
    else:
        ax1.text(0.5, 0.5, "Sin datos", ha='center')
    path_pie = _save_matplotlib_fig_to_temp(fig1)
    content_g1.append(ReportLabImage(path_pie, width=10*cm, height=6*cm))
    content_g1.append(Spacer(1, 10))
    story.append(KeepTogether(content_g1))

    # -- 3.2 Histograma --
    content_g2 = []
    content_g2.append(Paragraph("3.2. Histograma de Frecuencia de Tamaños", style_h3))
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    if diametros:
        ax2.hist(diametros, bins=15, color='#4DA6FF', edgecolor='black', alpha=0.7)
        ax2.set_xlabel("Tamaño (cm)", fontsize=8)
        ax2.set_ylabel("Frecuencia", fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
    path_hist = _save_matplotlib_fig_to_temp(fig2)
    content_g2.append(ReportLabImage(path_hist, width=12*cm, height=6*cm))
    content_g2.append(Spacer(1, 10))
    story.append(KeepTogether(content_g2))

    # -- 3.3 Curva --
    content_g3 = []
    content_g3.append(Paragraph("3.3. Curva Granulométrica (Pasante Acumulado)", style_h3))
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    if sorted_sizes:
        cumulative = [(i+1)/len(sorted_sizes)*100 for i in range(len(sorted_sizes))]
        ax3.plot(sorted_sizes, cumulative, marker='.', linestyle='-', color='#003366', linewidth=1)
        ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='P80')
        ax3.axhline(y=50, color='g', linestyle='--', alpha=0.5, label='P50')
        ax3.set_xlabel("Tamaño (cm)", fontsize=8)
        ax3.set_ylabel("% Pasante", fontsize=8)
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(fontsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
    path_curve = _save_matplotlib_fig_to_temp(fig3)
    content_g3.append(ReportLabImage(path_curve, width=12*cm, height=7*cm))
    content_g3.append(Spacer(1, 12))
    story.append(KeepTogether(content_g3))

    # 6. ANÁLISIS GEOMECÁNICO (Open Pit - Cielo Abierto)
    content_eval = []
    content_eval.append(Paragraph("4. EVALUACIÓN TÉCNICA Y RECOMENDACIONES", style_h2))
    
    pct_sobredim = stats.get('porcentaje_sobredim', 0)
    
    # --- VIÑETAS TÉCNICAS (MINERÍA CIELO ABIERTO) ---
    bullets = []
    
    # 1. Diagnóstico Geomecánico
    if st.session_state.esponjamiento_calc and st.session_state.esponjamiento_calc > 0:
        densidad_suelta = res['params'].get('densidad',0)/(1+st.session_state.esponjamiento_calc/100)
        bullets.append(f"<b>Diagnóstico Geomecánico:</b> El esponjamiento calculado del <b>{st.session_state.esponjamiento_calc:.2f}%</b> deriva en una densidad aparente de pila de <b>{densidad_suelta:.2f} g/cm³</b>. Este parámetro afecta directamente el <b>factor de llenado del balde (Fill Factor)</b> en las Palas Eléctricas/Hidráulicas y Cargadores Frontales.")
    
    # 2. Análisis de Fragmentación (Condicional - Open Pit)
    if pct_sobredim > 15:
        bullets.append(f"<b>Estado de Fragmentación (CRÍTICO):</b> Se detecta un <b>{pct_sobredim:.1f}%</b> de material sobredimensionado (bolonería), excediendo el límite operativo estándar. Esto impactará en la productividad de las <b>Palas de Carguío</b> (mayor tiempo de ciclo) y reducirá el throughput en el <b>Chancado Primario</b>.")
        bullets.append("<b>Diseño de Malla de Perforación:</b> Revisar el Burden actual; un valor excesivo puede estar generando confinamiento. Verificar si la malla (cuadrada o triangular) es adecuada para la dureza de la roca en este banco.")
        bullets.append("<b>Control de Calidad (QA/QC):</b> Verificar la altura del taco (stemming) para evitar eyección prematura de gases y asegurar la ruptura en la parte superior del banco (Cresta).")
        bullets.append("<b>Pata del Banco (Toe):</b> Evaluar si la sobre-perforación (Sub-drilling) es suficiente para garantizar el corte a nivel de piso y evitar problemas de piso duro (Hard Toe) para los equipos de carguío.")
    elif pct_sobredim > 5:
        bullets.append(f"<b>Estado de Fragmentación (ACEPTABLE):</b> Con un <b>{pct_sobredim:.1f}%</b> de sobretamaño, la pila de material es apta para el carguío, aunque existe margen para optimizar la granulometría de alimentación a planta.")
        bullets.append("<b>Optimización Operativa:</b> Se sugiere mantener los parámetros de Burden y Espaciamiento, pero monitorear la secuencia de tiempos de retardo entre filas para mejorar el desplazamiento de la pila.")
    else:
        bullets.append(f"<b>Estado de Fragmentación (ÓPTIMO):</b> Se observa una fragmentación excelente con solo un <b>{pct_sobredim:.1f}%</b> de gruesos. Esta condición maximiza la eficiencia de llenado de los Camiones Mineros y reduce el consumo energético en Chancado.")
        bullets.append("<b>Oportunidad de Ahorro:</b> Dado el alto grado de fragmentación, es viable analizar una ampliación de la malla (Burden x Espaciamiento) para reducir el Factor de Carga (kg/m³) y optimizar costos de explosivos.")

    # 3. Recomendación General (Cierre)
    bullets.append("<b>Mantenimiento del Sistema:</b> Se recomienda calibrar la referencia visual antes de cada voladura para mantener la precisión del análisis en los distintos frentes de carguío.")

    # Agregamos las viñetas al story usando el estilo BulletPro
    for b in bullets:
        p = Paragraph(f"• {b}", style_bullet_pro)
        content_eval.append(p)
        content_eval.append(Spacer(1, 4))
    
    story.append(KeepTogether(content_eval))

    # Pie de página final
    story.append(Spacer(1, 20))
    story.append(Paragraph("Generado automáticamente por SIFRAQ - Sistema Inteligente de Fragmentación Quinde", ParagraphStyle('footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
    buffer.seek(0)
    return buffer


# ... (Todo tu código anterior) ...

def show_report():
    st.markdown("## 📑 Reporte Técnico Final")
    st.markdown("Genera y descarga el reporte profesional. También puedes llevar el análisis al campo.")

    # --- URL DE TU APP (ACTUALIZAR AL FINAL) ---
    # Cuando tengas el link de Streamlit Cloud (ej: https://sifraq.streamlit.app), pégalo aquí.
    APP_URL = "https://share.streamlit.io/" 
    # -------------------------------------------

    pdf_buffer = crear_pdf_profesional()
    
    if pdf_buffer:
        st.markdown("---")
        
        # Usamos columnas para organizar: Descargas a la izquierda, QR a la derecha
        col_descargas, col_qr = st.columns([1.5, 1])
        
        with col_descargas:
            st.subheader("📥 Descargar Datos de Análisis")
            
            res = st.session_state.analysis_results
            
            # 1. Botón CSV
            if res and 'df' in res:
                csv = res["df"].to_csv(index=False)
                st.download_button(
                    "📄 Descargar CSV de Fragmentos",
                    data=csv,
                    file_name="datos_fragmentacion.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.write("") # Espacio vertical
            
            # 2. Botón PDF
            st.download_button(
                "📕 Descargar Informe PDF",
                data=pdf_buffer,
                file_name=f"Reporte_SIFRAQ_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.write("") # Espacio vertical
            
            # 3. Botón Imagen
            if res and 'img' in res:
                img_pil = Image.fromarray(cv2.cvtColor(res["img"], cv2.COLOR_BGR2RGB))
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                st.download_button(
                    "🖼️ Descargar Imagen Analizada",
                    data=img_buffer.getvalue(),
                    file_name="fragmentacion_analizada.png",
                    mime="image/png",
                    use_container_width=True
                )

        with col_qr:
            # --- SECCIÓN DEL QR EN LA INTERFAZ ---
            st.markdown("### 📱 Versión Móvil")
            st.info("Escanea para continuar en el frente de trabajo:")
            
            # Generar QR en memoria
            qr = qrcode.QRCode(version=1, box_size=8, border=2)
            qr.add_data(APP_URL)
            qr.make(fit=True)
            img_qr = qr.make_image(fill_color="black", back_color="white")
            
            # Convertir a bytes para mostrar en Streamlit
            img_qr_byte = io.BytesIO()
            img_qr.save(img_qr_byte, format='PNG')
            
            # Mostrar imagen centrada
            st.image(img_qr_byte, width=200, caption="Acceso Web Móvil SIFRAQ")

    st.markdown("---")
    if st.button("🔄 Realizar Nuevo Análisis", use_container_width=True):
        reset_app()

# ==========================================
# RUTEO PRINCIPAL
# ==========================================
sidebar_menu()

if st.session_state.step == 0: show_landing()
elif st.session_state.step == 1: show_project_data()
elif st.session_state.step == 2: show_upload()
elif st.session_state.step == 3: show_calibration()
elif st.session_state.step == 4: show_parameters()
elif st.session_state.step == 5: show_results()
elif st.session_state.step == 6: show_report()