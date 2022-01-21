from os import listdir
from os.path import isfile, join
import sys

data_type = sys.argv[1]

mypath = './datasets/dataset{}/final_batches/'.format(data_type)
csv_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
csv_files = [x for x in csv_files if '._' not in x]

targetpath =  './datasets/dataset{}/final_batch/'.format(data_type)
count = 1
for my_file in csv_files:
    txt_file = open(targetpath + 'final_' + data_type + str(count) +'.csv', 'w')
    text = ''
    # print(mypath + my_file)
    with open(mypath + my_file) as f:
        lines = f.readlines()

    for line in lines:
        l = line.strip()
        l = l.replace('"', '')
        text += l + '\n'

    text = text.strip()
    txt_file.write(text)
    txt_file.close()
    count += 1