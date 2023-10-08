from setuptools import setup, find_packages

setup(
    name='models',
    version='0.1.0',
    description='A simple example package',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'pandas'],
)