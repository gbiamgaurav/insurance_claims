
from setuptools import find_packages, setup 
from typing import List


HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List:
    
    """ This function will return the list of requirements """
    
    
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

    # test
        

setup(
    name="insurance_claims",
    version="0.0.1",
    author="Gaurav",
    author_email="gaurav.bhattacharya10@gmail.com",
    packages=find_packages(),
    include_dirs=get_requirements("requirements.txt")
)