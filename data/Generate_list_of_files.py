#get a folder and put their names in a list

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("ut/test_xml/") if isfile(join("ut/test_xml/", f))]

textfile = open("ut/test.txt", "w")
for element in onlyfiles:
    textfile.write(element + "\n")
textfile.close()