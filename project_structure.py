import os
from pathlib import Path

project_name="src"

list_of_files = [
	f"{project_name}/__init__.py",
	f"{project_name}/component/__init__.py",
	f"{project_name}/component/dataingestion.py",
	f"{project_name}/component/datatransformation.py",
	f"{project_name}/component/datavalidation.py",
	f"{project_name}/component/modeltraining.py",
	f"{project_name}/component/datacleaning.py",
	f"{project_name}/logger.py",
	f"{project_name}/utils.py",
	f"{project_name}/exception.py",
	f"{project_name}/predict_pipeline/__init__.py",
	f"{project_name}/predict_pipeline/predict_pipeline.py",
]

for path in list_of_files:
	filepath=Path(path)

	filedir,filename=os.path.split(path)
	if filedir!="":
		os.makedirs(filedir, exist_ok=True)
	
	
	if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
		with open(filepath, "w") as f:
			pass
