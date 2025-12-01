from setuptools import setup, find_packages

setup(
    name="sifraq-app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "Pillow==10.2.0",
        "scipy==1.11.4",
        "plotly==5.18.0",
        "reportlab==4.1.0",
        "qrcode[pil]==7.4.2",
        "streamlit==1.35.0",
        "streamlit-drawable-canvas==0.9.3",
        "opencv-python-headless==4.9.0.80",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "matplotlib==3.7.5",
        "pycocotools==2.0.7",
    ]
)