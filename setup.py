from setuptools import setup, find_packages

setup(
    name="air-quality-forecasting-alert-system",
    version="1.0.0",
    author="Khushi Yadav",
    author_email="ky19tech@gmail.com",
    description="CPCB/OpenAQ Air Quality Forecasting (24h-72h) and Automated Telegram/Email Alert System for Delhi & Lucknow",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md").readable() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/khushiyadav35/air-quality-forecasting-alert-system",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "streamlit>=1.30.0",
        "requests>=2.31.0",
    ],
)
