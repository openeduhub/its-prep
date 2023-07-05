#!/usr/bin/env python3
from setuptools import setup

setup(
    name="nlprep",
    version="0.0.1",
    description="A collection of NLP tools for text-preprocessing tasks.",
    packages=["nlprep"],
    install_requires=[
        d for d in open("requirements.txt").readlines() if not d.startswith("--")
    ],
    package_dir={"": "."},
)
