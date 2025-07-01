import roboflow
roboflow.login()

rf = roboflow.Roboflow()
project = rf.workspace("team-roboflow").project("coco-128")  # Replace with your project name
dataset = project.version(2).download("coco")  # Replace with your version number
