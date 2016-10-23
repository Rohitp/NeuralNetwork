from scipy.special import expit
import numpy as np
import csv
import sys



INPUT_NODES = 784
OUTPUT_NODES = 10
HIDDEN_NODES = 100 #Only Works for a single layer of hidden nodes. Maybe use a list instead?

LEARNING_RATE = 0.3

# ascii progress bar. Not relevant to the network
# stolen from top answer at stack overflow.
# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class NeuralNetwork:
    def __init__(self, inputNodes, outputNodes, hiddenNodes, learningRate):
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.hiddenNodes = hiddenNodes
        self.learningRate = learningRate
        self.activationFunction = expit

        self.totalNodes =  inputNodes + outputNodes + hiddenNodes

        # normal distribution for with mean 0 and SD as inverse square root of number of links
        self.inputHiddenWeights = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.outputHiddenWeights = np.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes , self.hiddenNodes))

    def query(self, inputs):

        inputs = np.array(inputs, ndmin = 2).transpose()

        hiddenInputs = np.dot(self.inputHiddenWeights, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        # calculate for hidden to output layer
        finalInputs = np.dot(self.outputHiddenWeights, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        return finalOutputs

    def train(self, inputs, targets):

        # transposing inouts for matrix multiplication. Using ndmin = 2 cause lists can't be transposed
        inputs = np.array(inputs, ndmin = 2).transpose()
        targets = np.array(targets, ndmin=2).transpose()

        hiddenInputs = np.dot(self.inputHiddenWeights, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = np.dot(self.outputHiddenWeights, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        # output layer error
        outputErrors = targets - finalOutputs

        # hidden layer error is the output layer error split by weight and recombined at
        # specific node for all it's paths

        hiddenErrors = np.dot(self.outputHiddenWeights.transpose(), outputErrors)

        # uodate weights for links between hidden and output layers
        self.outputHiddenWeights += self.learningRate * np.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), hiddenOutputs.transpose())

        # update between hidden and input layers
        self.inputHiddenWeights += self.learningRate * np.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), inputs.transpose())

        # print(inputs)





neuralNetwork = NeuralNetwork(INPUT_NODES, OUTPUT_NODES, HIDDEN_NODES, LEARNING_RATE)

targets = []
trainingInputs = []

print("Reading file... ")
with open("mnist_train.csv") as trainingFile:
    line = csv.reader(trainingFile)
    for i, row in enumerate(line):
        targets.append(row[0])
        trainInp = [((int(x) / 255) * 0.99) + 0.01 for x in row[1:]]
        trainingInputs.append(trainInp)

print("\n")
printProgress(0, len(trainingInputs), prefix = 'Training Progress:', suffix = 'Complete', barLength = 50)
for i, trainingInput in enumerate(trainingInputs):
    target = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    target[int(targets[i])] = 0.99
    neuralNetwork.train(trainingInput, target)
    printProgress(i, len(trainingInputs), prefix = 'Training Progress:', suffix = 'Complete', barLength = 50)


# test with a certain number from the test set
TEST_LIMIT = 20
print("\n")
with open("mnist_test.csv") as testFile:
    line = csv.reader(testFile)
    for i, row in enumerate(line):
        if i > TEST_LIMIT:
            break
        testInput = [((int(x) / 255) * 0.99) + 0.01 for x in row[1:]]
        result = neuralNetwork.query(testInput).tolist()
        maxValue = max(result)
        index = result.index(maxValue)
        print("The predicted number is <"+str(index)+"> the correct answer is <"+row[0]+">")
