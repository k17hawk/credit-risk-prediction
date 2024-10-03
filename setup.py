"""
author: @kumar dahal
This function is written for version controlling.
It will read long description from readme.md file.
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Set the initial version
__version__ = "0.0.0"

REPO_NAME = "credit-risk-prediction"
AUTHOR_USER_NAME = "kumar dahal"
SRC_REPO = "credit_risk"
AUTHOR_EMAIL = "kumardahal536@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Project for credit status prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Corrected here
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    packages=setuptools.find_packages(),  # This will find packages in the whole project
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
