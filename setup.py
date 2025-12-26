#setup.py we used to create our machinelearning/mlproject/anyother project as Package.so that we can deploy it.
# We can utilize this deployed package anywhere

from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        # If requirements.txt file have -e. ,just remove.because -e. is reference to setup.py
        #but here we are running setup.py,so we don't need -e.
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Uday',
author_email='udayasankar.jalli@gmail.com',
packages=find_packages(), # it will check how many folders having __init__.py
install_requires=get_requirements('requirements.txt')

)