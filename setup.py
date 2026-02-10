"""
This setup.py is responsible for creating the ML application as a package.
"""


from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements.
    """

    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements if req != "-e ."]
    return requirements


setup(
    name="ml_project",
    version="0.0.1",
    author="Krish Vardhan Pal",
    author_email="krishvardhan9369@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
