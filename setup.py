from setuptools import find_packages,setup
from typing import List

hyphen_edot = '-e .'
def get_requirments(filepath:str)->List[str]:
    # this func will return the names of the packges from requirments.txt 
    req = []
    with open(filepath) as f:
        pack = f.readlines()
        req = [name.replace('\n','') for name in pack]

        if hyphen_edot in req:
            req.remove(hyphen_edot)
    
    return req

setup(
    name='ML_Project',
    author="Subhadip Mudi",
    version="0.0.1",
    author_email="subhadipmudi2020@gmail.com",
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')
)