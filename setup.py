from setuptools import find_packages, setup

setup(
    name='blocks.extras',
    namespace_packages=['blocks'],
    install_requires=['blocks'],
    packages=find_packages(),
    zip_safe=False
)
