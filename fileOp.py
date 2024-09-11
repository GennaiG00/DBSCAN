import csv
def readFromFile(fileName):
    with open(fileName, mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    return data
