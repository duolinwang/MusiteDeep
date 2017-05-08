from setuptools import setup
from setuptools import find_packages


setup(name='MusiteDeep',
      version='0.0.1',
      description='MusiteDeep:a deep learning framework for general and kinase-specific phosphorylation site prediction',
      author='Duolin Wang',
      author_email='deepduoduo@gmail.com',
      url='https://../../',
      download_url='https://../..',
      license='GNU2.0',
      install_requires=['theano>=0.8.0', 'pyyaml', 'keras==1.1.0','numpy>=1.8.0','h5py','pandas'],
      extras_require={
          'h5py': ['h5py'],
      },
      packages=find_packages(),
      package_data = {
      'MusiteDeep' : ['MusiteDeep/models/*']
      }
      )
