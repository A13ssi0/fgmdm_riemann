from setuptools import setup, find_packages


setup(name='fgmdm_riemann',
      packages=find_packages(),
        install_requires=['numpy','scipy','joblib','pyriemann','scikit-learn','pandas','tqdm'],
        version='0.1',
        description='Riemannian classifier for EEG data',
        author='Alessio Palatella',
      )
