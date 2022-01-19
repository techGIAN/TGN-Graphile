from os import listdir
from os.path import isfile, join

mypath = './datasets/datasetB/final_batches/'
csv_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
csv_files = [x for x in csv_files if '._' not in x]

targetpath =  './datasets/datasetB/final_batch/'
count = 1
for my_file in csv_files:
    txt_file = open(targetpath + 'final_B' + str(count) +'.csv', 'w')
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