#!/bin/bash
# setup.sh - Script de instalación para Streamlit Cloud

echo "Instalando dependencias del sistema..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

echo "Instalando dependencias de Python..."
pip install --upgrade pip
pip install -r requirements.txt

# Intentar instalar detectron2 si no está presente
python -c "import detectron2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalando detectron2 desde source..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
fi

echo "Instalación completada."