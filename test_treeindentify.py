from treeindentify import treeArea
from treeindentify import knnClassifier
from treeindentify import svmClassifier
from treeindentify import mapFeature
from treeindentify import sigmoid, costFunctionReg, costFunctionGrad, kmeanColor
#from scipy.optimize import fmin_bfgs, fmin
#import cv2
#from matplotlib import pyplot as plt
import numpy as np

##training knn classifier
fileName = ["green" + str(x) + ".jpg" for x in range(1, 7)]
testName = ["test" + str(y) + ".jpg" for y in range(1, 11)]
#labels = np.repeat([1, 0], (trainData.shape[0] / 2))
#labels = labels.reshape([trainData.shape[0], 1]).astype(np.float32)
#labels_test = np.repeat([1, 0], (testData.shape[0] / 2))
#labels_test = labels_test.reshape([testData.shape[0], 1]).astype(np.float32)
knn = knnClassifier(fileName, testName, save = 0)

knn.autoSelect(datatype = 1, bgr2hsv = 0)
knn.readFile(datatype = 0, bgr2hsv = 0)
knn.train()
knn.test()

svm = svmClassifier(fileName, testName, save = 0)
svm.autoSelect(datatype = 1, bgr2hsv = 0)
svm.readFile(datatype = 0, bgr2hsv = 0)
svm.train()
svm.test()



##running the main script
lower_tree = np.array([30, 105, 35])
upper_tree = np.array([60, 205, 100])
pictype = ".jpg"
fileName = [str(x) + pictype for x in range(1, 28)]
saveName = [str(x) + "_svm_knnbj" + ".jpg" for x in range(1, 28)]
test = treeArea(lower_tree, upper_tree, pictype)
#test.readFiles(fileName)
#test.knnMask(knn)
#test.calcMask(medianblur = 0, close = 0, after = 0)
#test.showPic(fileName, select = "mask")
#test.calcMask(medianblur = 0, close = 1, after = 0)

#test.calcGreenArea()
test.run(fileName, typeof = "svm", classifier = svm)
#test.showPic(fileName, select = "mask")
test.savePic(saveName, select = "mask")

#temp = knn.trainData
#temp_h = mapFeature(temp[:,0], temp[:,1], temp[:,2])
#temp_h = (temp_h - np.min(temp_h)) / (np.max(temp_h) - np.min(temp_h))
#temp_h *= 1
#labels_h = np.repeat([1, 0], (temp_h[0].shape[0] / 2))
#labels_h.shape = [labels_h.size, 1]
#theta = np.zeros([temp_h[0].shape[1], 1])
#lamd = 1
#j = costFunctionReg(theta, temp_h[0], labels_h, lamd)
#grad = costFunctionGrad(theta, temp_h[0], labels_h, lamd)
