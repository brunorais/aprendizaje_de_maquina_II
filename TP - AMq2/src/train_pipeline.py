import subprocess

subprocess.run(['Python', './src/feature_engineering.py', 'train'])

subprocess.run(['Python', './src/train.py'])
