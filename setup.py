from setuptools import setup, find_packages

setup(
    name="fdaft",
    version="1.0.0",
    description="Fast Double-Channel Aggregated Feature Transform for Matching Planetary Remote Sensing Images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="FDAFT Team",
    author_email="fdaft@example.com",
    url="https://github.com/username/fdaft",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-contrib-python>=4.5.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-image>=0.17.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "jupyter>=1.0.0",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
