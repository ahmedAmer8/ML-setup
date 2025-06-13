# build the ML project as a package

from setuptools import setup, find_packages
from typing import List


HYPENATED_PACKAGE = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """
    This function reads the requirements file and returns a list of packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    
    # Remove any whitespace characters like `\n` at the end of each line
    requirements = [req.strip() for req in requirements if req.strip()]
    
    if HYPENATED_PACKAGE in requirements:
        requirements.remove(HYPENATED_PACKAGE)
    
    return requirements


setup(
    name='ml_project',
    version='0.1',
    author='Ahmed_Amer',
    author_email='ahmed.mohammad.amer@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)