import os, glob
import librosa

path_dir = "../download_audioset/audiosetdata/"
dirs = os.listdir(path_dir)

for dir in dirs:
    files = os.listdir(path_dir+dir)
    if(len(files)>700):
        print(dir, len(files))