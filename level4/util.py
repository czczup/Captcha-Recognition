def read_CSV(file_path, keyIndex=0, valueIndex=1):
    """ This is a method to read CSV files. """
    dataDict = {}
    with open(file_path, "r") as csvFile:
        dataLine = csvFile.readline().strip("\n")
        while dataLine != "":
            tmpList = dataLine.split(',')
            dataDict[int(tmpList[keyIndex])] = tmpList[valueIndex]
            dataLine = csvFile.readline().strip("\n")
        csvFile.close()
    return dataDict