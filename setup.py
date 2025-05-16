from setuptools import setup, find_packages

setup(
    name="face-recognition",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "deepface>=0.0.79",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
) 