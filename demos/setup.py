# This function is to deal with the dependence linear regression module needs.

import os
import sys

def find_root(repo="fedlearn-algo"):
    root = os.getcwd()
    folders = root.split('/')
    last = len(folders)-1
    while last>=0:
        if folders[last].lower()==repo.lower():
            break
        else:
            last -= 1
    if last<0:
        print("Error: Root folder not found!")
        return None
    else:
        rootFolder = ""
        for ii in range(0, last):
            rootFolder += folders[ii]
            rootFolder += '/'
        rootFolder += folders[last]
        return rootFolder

def deal_with_path(repo="fedlearn-algo"):
    rootFolder = find_root(repo)
    sys.path.append(rootFolder)

if __name__=="__main__":
    print(find_root())