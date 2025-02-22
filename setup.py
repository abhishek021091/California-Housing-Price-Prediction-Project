from setuptools import find_packages,setup 
from typing import List

def get_requirements()->List[str]:
	"""
	This function will return list of requirements
	"""
	requirement_lst = []
	try:
		with open('requirements.txt','r') as files:
			#Read lines from the files
			lines = files.readlines()
			## Process each line
			for line in lines:
				requirement=line.strip()
				## ignore empty lines and -e .
				if requirement and requirement!='-e .':
					requirement_lst.append(requirement)
	except FileNotFoundError:
		print("requirement.txt file not found")

	return requirement_lst

setup(
	name="California Housing Price Prediction",
	version="0.0.1",
	author="Abhishek Singh",
	author_email="abhishek2001singh2001@gmail.com",
	packages=find_packages(),
	install_requires=get_requirements()
)