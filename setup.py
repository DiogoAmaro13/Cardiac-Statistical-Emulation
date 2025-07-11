from setuptools import setup, find_packages

setup(
    name="gpr_modelling",
    version="0.1.0",
    author="Diogo Amaro",
    description="Gaussian Process Regression Toolkit for Cardiac Statistical Emulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

