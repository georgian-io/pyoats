from os import listdir
from os.path import isfile, join

def get_files_from_path(path: str):
    return sorted([f for f in listdir(path) if isfile(join(path, f))])
