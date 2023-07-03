#!/usr/bin/env python3
from setuptools import setup

setup(
    name="wloprep",
    version="0.0.1",
    description="A collection of NLP text-preprocessing tasks for the context of WLO",
    packages=["wloprep"],
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    package_dir={"": "."},
)
