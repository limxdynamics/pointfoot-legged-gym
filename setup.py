from setuptools import find_packages
from distutils.core import setup

setup(
    name="legged_gym",
    version="1.0.0",
    author="Hongxi Wang, Nicolas Rudin",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="rudinn@ethz.ch",
    description="Modified from Isaac Gym environments for Legged Robots",
    install_requires=["isaacgym", "matplotlib"],
)
