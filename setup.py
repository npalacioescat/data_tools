from setuptools import setup, find_packages

setup(name='data_tools',
      version='0.0.3',
      description="Collection of Python functions and classes designed to make a Computational Biologist's life easier.",
      url='https://github.com/Nic-Nic/data_tools',
      author='Nicolas Palacio',
      author_email='nicolaspalacio91@gmail.com',
      license='GNU-GLPv3',
      package_dir = {'':'src'},
      packages=find_packages('src'),
      zip_safe=False)
