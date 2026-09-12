import glob
import os

for f in glob.glob('ml-scripts/*.py'):
    with open(f, 'r') as file:
        content = file.read()
    
    content = content.replace('Path(__file__).resolve().parent', 'Path(__file__).resolve().parent.parent')
    
    with open(f, 'w') as file:
        file.write(content)
print("Updated all scripts.")
