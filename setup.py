from setuptools import setup, find_packages

setup(
    name='lma_diagnostic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'typer', 'loguru', 'tqdm', 'numpy', 'pandas', 
        'scikit-learn', 'opencv-python', 'scipy', 
        'scikit-image', 'matplotlib', 'flask', 
        'flask-cors', 'joblib', 'werkzeug'
    ],
    python_requires='>=3.10',
    author='Your Name',
    description='LMA Diagnostic AI Application',
)
