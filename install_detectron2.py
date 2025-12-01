# install_detectron2.py
import subprocess
import sys
import os

def install_detectron2():
    """Instala detectron2 desde GitHub con flags específicos"""
    print("Instalando PyTorch primero...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    print("Instalando dependencias de detectron2...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "pycocotools==2.0.7"
    ])
    
    print("Clonando e instalando detectron2...")
    # Clonar repositorio
    subprocess.check_call([
        "git", "clone", "https://github.com/facebookresearch/detectron2.git"
    ])
    
    os.chdir("detectron2")
    
    # Instalar en modo desarrollo (evita compilación compleja)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-e", ".",
        "--no-build-isolation"
    ])
    
    print("✅ Detectron2 instalado exitosamente")

if __name__ == "__main__":
    install_detectron2()