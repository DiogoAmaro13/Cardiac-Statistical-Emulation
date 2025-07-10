from setuptools import setup, find_packages

setup(
    name="gpr_modelling",
    version="0.1",
    description="Gaussian Process Regression Modeling Toolkit",
    author="Diogo Amaro",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # Add required dependencies here later, like 'numpy', 'scikit-learn', etc.
)
