import os
import shutil

src_dir = '../TNC/path/'
dest_dir = '.'

for root, dirs, files in os.walk(src_dir):
    #print("find the path:")
    for file in files:
        if file == 'circuit.txt':
            #print("find the path:",root[-8:])
            src_file_path = os.path.join(root,file)
            #print(src_file_path)
            circuits_name = root[-8:]

            circuits_dir = os.path.join(dest_dir,circuits_name)
            #print(circuits_dir)

        
            os.makedirs(circuits_dir,exist_ok=True)

            shutil.copy2(src_file_path,circuits_dir)

