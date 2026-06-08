from setuptools import setup, find_packages

setup(
    name="onko-medical-image-classification",
    version="1.0.0",
    author="Your Name",
    description="Medical image classification for PCam, NCT-CRC-HE, and ISIC datasets",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "h5py>=3.9.0",
        "datasets>=2.14.0",
    ],
    python_requires=">=3.8",
)
