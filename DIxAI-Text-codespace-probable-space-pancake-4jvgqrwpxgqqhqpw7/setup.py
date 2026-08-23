from setuptools import setup, find_packages

setup(
    name="dixai",
    version="0.1.0",
    description="Decision-Information Explainable AI Framework",
    author="Haythem Ghazouani",
    author_email="haythem.ghazouani@enicar.u-carthage.tn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "torch>=2.0.0",
        "scipy",
        "scikit-learn",
        "pandas",
        "tqdm",
        "matplotlib",
        "torchvision",
    ],
)
