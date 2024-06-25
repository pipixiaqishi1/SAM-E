
"""
Setup of SAM-E
Author: Junjie Zhang
"""
from setuptools import setup, find_packages

requirements = [
    "numpy",
    "scipy",
    "einops",
    "pyrender",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow",
    "pyquaternion",
    "matplotlib",
    "clip @ git+https://github.com/openai/CLIP.git",
    "icecream"
]

__version__ = "0.0.1"
setup(
    name="samE",
    version=__version__,
    description="SAM-E",
    long_description="",
    author="Junjie Zhang",
    url="",
    keywords="robotics,computer vision",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
    ],
    packages=['samE'],
    install_requires=requirements,
)
