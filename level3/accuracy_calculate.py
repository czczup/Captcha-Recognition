from util import read_CSV
import conf


def accuracy_calculate():
    # load mappings.txt
    dict1 = read_CSV(conf.MAPPINGS)
    dict2 = read_CSV(conf.TEST_MAPPINGS)

    # Calculate the accuracy.
    correct = 0
    for i in range(conf.TEST_NUMBER):
        if dict1[i]==dict2[i]:
            correct += 1
        else:
            print("Number:", i, "False:", dict1[i], "True:", dict2[i])

    print("Accuracy:", correct/conf.TEST_NUMBER)
    return correct/conf.TEST_NUMBER


if __name__=='__main__':
    accuracy_calculate()
