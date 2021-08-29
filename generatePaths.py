import os
import readData as rd 

if __name__ == "__main__":

    ids = rd.generateIDs()
    if not os.path.exists("img"):
        os.mkdir("img")
    for id0 in ids:
        dirName = "img/sub-{}".format(id0) 
        if not os.path.exists(dirName):
            os.mkdir(dirName)