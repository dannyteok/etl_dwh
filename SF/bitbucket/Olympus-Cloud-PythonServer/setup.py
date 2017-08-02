from distutils.core import setup

setup(
    name='smartfocus-olympus-cloud-pythonserver',
    version='0.1.0',
    packages=['tests', 'SOMTrainer', 'SOMTrainer.training', 'SOMTrainer.validation'],
    url='git@bitbucket.org:smartfocus1/smartfocus-olympus-cloud-pythonserver.git',
    license='Proprietary',
    author='Panagiotis Agis Oikonomou Filandras',
    author_email='agis.oikonomou@smartfocus.com',
    description='SOM Training Server',
    install_requires=['joblib','numpy','pandas','scikit-learn',
                      'SmartFocus-Olympus-RnD-DataProcessing',
                      'SmartFocus-Olympus-RnD-SOM',
                      'twisted','jsonschema']
)
