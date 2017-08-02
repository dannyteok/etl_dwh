from distutils.core import setup

setup(
    name='SmartFocus-Olympus-RnD-DataProcessing',
    version='0.1.0',
    packages=['processing', 'processing.utils'],
    url='git@bitbucket.org:smartfocus1/smartfocus-olympus-rnd-dataprocessing.git',
    license='Proprietary',
    author='Panagiotis Agis Oikonomou Filandras',
    author_email='agis.oikonomou@smartfocus.com',
    description='Parser Library for BLE Data',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn','pymongo', 'joblib','behave','pytest']
)
