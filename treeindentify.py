import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
##################################logistic-dev##################################
def sigmoid(x):
    g = np.zeros(shape = x.shape)
    g = .5 * (1 + np.tanh(.5 * x))
    return g

#mapFeature:map feature space to higher dimension,
#this function here ony map 3 dim space to
#high feature space
#examples:degree = 1:no mapping;degree = 2,mapping to 2 degree polynomial space
#!BEAWARE!:when temp = (i, j, k), higher predict accuracy find in octave protocol
#!Warning!:divide by zero encounterd in log with current data set,precision loss
def mapFeature(X1, X2, X3, degree = 1):
    d = degree + 1
    out = np.zeros([X1.shape[0], 0])
    count = 0
    for i in np.arange(0, d):
        for j in np.arange(0, d-i):
            for k in np.arange(0, d-i-j):
                temp = np.power(X1, k) *np.power(X2, j) * np.power(X3, i)
                temp = temp.reshape([temp.shape[0], temp.size / temp.shape[0]])
                out = np.hstack((out, temp))
                count += 1
    return out, count

def costFunctionReg(theta, X, labels, lamd):
    m = X.shape[0]
    theta = theta.reshape([len(theta), 1])
    temp1 = np.log(sigmoid(np.matrix(X) * np.matrix(theta)))
    temp1 = -1 * labels * np.array(temp1)
    #temp2 = 0.5
    temp2 = np.log(1 - sigmoid(np.matrix(X) * np.matrix(theta)))
    temp2 = (1 - labels) * np.array(temp2)
    reg = np.sum(np.square(theta[1:(len(theta)+1)])) * (lamd / (2*m))
    j = np.sum(temp1 - temp2) / m + reg
    return j

def costFunctionGrad(theta, X, labels, lamd):
    m = X.shape[0]
    theta = theta.reshape([len(theta), 1])
    grad = np.zeros([X.shape[1], 1]).astype(np.float32)
    for i in np.arange(0, len(theta)):
        if (i == 0):
            t2 = X[:, i].reshape([m, 1])
            t1 = sigmoid(np.matrix(X) * np.matrix(theta)) - labels
            t1 = np.array(t1)
            grad[i, 0] = np.sum(t1 * t2) / m
        else:
            t2 = X[:, i].reshape([m, 1])
            t1 = sigmoid(np.matrix(X) * np.matrix(theta)) - labels
            t1 = np.array(t1)
            grad[i, 0] = (np.sum(t1 * t2) / m) + (theta[i, 0] * lamd / m)
   
    return grad.flatten()

def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(shape = [m, 1])
    p = sigmoid(np.matrix(X) * np.matrix(theta))
    p = np.array(p)
    p[ p >= 0.5] = 1
    p[ p < 0.5] = 0
    return p
###############################################################################

##################################kmeans-dev###################################
def kmeanColor(img, K=30, save = 0):
    img1 = cv2.imread(img)
    z = img1.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    flag = cv2.KMEANS_RANDOM_CENTERS
    compa, label, center = cv2.kmeans(z, K, criteria, 10, flag)

    width, height = 60, 30
    center = np.uint8(center)
    square = np.repeat(np.arange(0, K), width * height)
    color_K = center[square.flatten()]
    color_K = color_K.reshape([height*K, width, 3])
    if save == 1:
        cv2.imwrite(img+"kmean.jpg", color_K)
    cv2.imshow("K-mean Color", color_K)
    cv2.waitKey(0)
    return center


###############################################################################

####################################treeArea###################################
class treeArea:

    def __init__(self, lower, upper, pictype):
        self.type = pictype
        self.lower = lower
        self.upper = upper
        self.fileList = []
        self.maskList = []
        self.afterList = []
        self.result = []

    def readFiles(self, fileName, bgr2hsv = 1):
        print "\n###Reading files###\n"
        for f in tqdm(fileName):
            img = cv2.imread(f)
            if (bgr2hsv == 1):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.fileList.append(img)
            print("\n")

    

    # tag "pic":show original picture
    # tag "mask":show the mask of greearean
    # tag "res":show the result of picture after applying mask to the original picture

    def showPic(self, fileName, select = "pic"):
        index = 0
        if (select == "pic"):
            for pic in self.fileList:
                cv2.imshow(fileName[index], pic)
                cv2.waitKey(0)
                index += 1
        if (select == "mask"):
            for mask in self.maskList:
                cv2.imshow(fileName[index], mask)
                cv2.waitKey(0)
                index += 1
        if (select == "res"):
            for res in self.afterList:
                cv2.imshow(fileName[index], res)
                cv2.waitKey(0)
                index += 1

    def savePic(self, fileName, select = "mask"):
        index = 0
        print "\n###Saving results###\n"
        if (select == "pic"):
            for pic in self.fileList:
                pic = pic.astype(np.uint8)
                cv2.imwrite(fileName[index], pic)
                index += 1
        if (select == "mask"):
            for mask in self.maskList:
                mask = mask.astype(np.uint8)
                cv2.imwrite(fileName[index], mask)
                index += 1
        if (select == "res"):
            for res in self.afterList:
                res = res.astype(np.uint8)
                cv2.imwrite(fileName[index], res)
                index += 1
        print "\n!Done!"
            
        

    #calcMask(self, medianblur, close, after)
    # close : using cv2.MOPH_CLOSE to reduce noise and small dot in the graph
    # medianblur:making the edge of greenArea more smooth ,also reducing noise, but can not reduce
    # small dot in the green area
    # after:mask the original picture

    def calcMask(self, paraset = [0, 1, 0]):
        print "\n###Calculating Mask###\n"
        def trans(para):
            if (para == 0):
                return "False"
            else:
                return "True"
        paraset_t = map(trans, paraset)
        print "parameters of calcMask:\n"
        print "medianblur:",paraset_t[0]
        print "MORPH_CLOSE:",paraset_t[1]
        print "USE_MASK:",paraset_t[2]
    
        for pic in self.fileList:
            mask = cv2.inRange(pic, self.lower, self.upper)
            if (paraset[0] == 1):
                mask = cv2.medianBlur(mask, 3)
            if (paraset[2] == 1):
                res = cv2.bitwise_and(pic, pic, mask = mask)
                self.afterList.append(res)
            if (paraset[1] == 1):
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            self.maskList.append(mask)

    #knnMask(self, knnClassifier)
    #knnClassifier:the trained knn classifier
    #WARNING:bad result if the picture has large area of "dark green" like water, green
    #playground, to prevent this result, shoult build a better training set

    def knnMask(self, knnClassifier):
        knnC = knnClassifier
        print "\n###runing KNN classifier###\n"
        for pic in tqdm(self.fileList):
            print "\n"
            temp = pic.copy()
            temp = temp.reshape([temp.size / 3, 3]).astype(np.float32)
            ret, result, neighbours, dist = knnC.knn.find_nearest(temp, k = 5)
            result = result.reshape(pic.shape[0], pic.shape[1])
            result[result == 1] = 255
            result_c = result.copy()
            result_c = result_c.astype(np.uint8)
            #kernel = np.ones((3, 3), np.uint8)
            #result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            result = cv2.medianBlur(result, 3)
            self.maskList.append(result)
            res = cv2.bitwise_and(pic, pic, mask = result_c)
            self.afterList.append(res)

    def svmMask(self, svmClassifier):
        svmC = svmClassifier
        print "\n###running svm classifier###\n"
        for pic in tqdm(self.fileList):
            print "\n"
            temp = pic.copy()
            temp = temp.reshape([temp.size / 3, 3]).astype(np.float32)
            result_svm = svmC.svm.predict_all(temp)
            result_svm = result_svm.reshape(pic.shape[0], pic.shape[1])
            result_svm[result_svm == 1] = 255
            result_c = result_svm.copy()
            result_c = result_c.astype(np.uint8)
            result_svm = cv2.medianBlur(result_svm, 3)
            self.maskList.append(result_svm)
            res = cv2.bitwise_and(pic, pic, mask = result_c)
            self.afterList.append(res)
            

    def calcGreenArea(self, shape = "rectangle"):
        print "\n###Counting greenarea###\n"
        count = 0.0000 
        for mask in self.maskList:
            print (">"),
            #mask_t = mask.copy()
            #mask_t[mask_t > 0] = 1
            mask_t = np.where(mask > 0)
            if (shape == "circle"):
                mask_all = 3.14159 * (mask.shape[0] / 2) * (mask.shape[0] / 2)
                count = (len(mask_t[0]) * 1.000) / mask_all
            if (shape == "rectangle"):
                count = (len(mask_t[0]) * 1.000) / (mask.size * 1.000)
            else:
                pass
            count *= 100.0000
            #count = len(mask_t[0])
            self.result.append(count)
            
    

    def run(self, fileName, typeof = 0, classifier = 0):
        self.readFiles(fileName, bgr2hsv = 0)
        if (typeof == "bitwise"):
            self.calcMask(paraset = [0, 1, 1])
        if (typeof == "knn"):
            self.knnMask(classifier)
        if (typeof == "svm"):
            self.svmMask(classifier)
        self.calcGreenArea(shape = "rectangle")
        
        


class knnClassifier:

    def __init__(self, trainName, testName, save = 0):
        self.save = save
        self.trainName = trainName
        self.testName = testName
        self.trainData = np.empty(shape = [0, 3])
        self.testData = self.trainData.copy()
        self.knn = cv2.KNearest()

    def readFile(self, datatype = 1, bgr2hsv = 1):
        if (datatype == 1):
            for f in self.trainName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape([(temp.size / 3), 3])
                self.trainData = np.append(self.trainData, temp)
                self.trainData = self.trainData.reshape([(self.trainData.size / 3), 3])
                self.trainData = self.trainData.astype(np.float32)
            #self.trainData = mapFeature(self.trainData[:,0],self.trainData[:,1],
                                        #self.trainData[:,2]).astype(np.float32)
            
        if (datatype == 0):
            for f in self.testName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape([(temp.size / 3), 3])
                self.testData = np.append(self.testData, temp)
                self.testData = self.testData.reshape([(self.testData.size / 3), 3])
                self.testData = self.testData.astype(np.float32)
            #self.testData = mapFeature(self.testData[:,0],self.testData[:,1],
                                        #self.testData[:,2]).astype(np.float32)

    #use kmean to auto select color
    def autoSelect(self, K = 30, datatype = 1, bgr2hsv = 1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        flag = cv2.KMEANS_RANDOM_CENTERS
        
        if (datatype == 1):
            for f in self.trainName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape((-1, 3)).astype(np.float32)
                compa_f, label_f, center_f = cv2.kmeans(temp, K, criteria, 10, flag)
                self.trainData = np.vstack((self.trainData, center_f))
            self.trainData = self.trainData.astype(np.float32)
        if (datatype == 0):
            for f in self.testName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape((-1, 3)).astype(np.float32)
                compa_f,label_f,center_f = cv2.kmeans(temp, K, criteria, 10, flag)
                self.testData = np.vstack((self.testData, center_f))
            self.testData = self.testData.astype(np.float32)
                
    def train(self):
        print "\n###training KNN classifier....###\n"
        print "training data shape:", self.trainData.shape
        trainLabels = np.repeat([1, 0], (self.trainData.shape[0] / 2))
        trainLabels = trainLabels.reshape([self.trainData.shape[0], 1]).astype(np.float32)
        self.knn.train(self.trainData,trainLabels)
        print"....!Done!....", self.knn

    def test(self):
        print "\n###testing KNN classifier....###\n"
        print "test data shape:", self.testData.shape
        testLabels = np.repeat([1, 0], (self.testData.shape[0]/2))
        testLabels = testLabels.reshape([self.testData.shape[0], 1]).astype(np.float32)
        ret, result, neighbours, dist = self.knn.find_nearest(self.testData, k = 5)
        matches = result == testLabels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print "matches:",correct,"/",self.testData.shape[0]
        print "accuracy:", accuracy


class svmClassifier(knnClassifier):

    def __init__(self, trainName, testName, save = 0):
        self.save = save
        self.trainName = trainName
        self.testName = testName
        self.trainData = np.empty(shape = [0, 3])
        self.testData = self.trainData.copy()
        self.svm = cv2.SVM()

    def readFile(self, datatype = 1, bgr2hsv = 1):
        if (datatype == 1):
            for f in self.trainName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape([(temp.size / 3), 3])
                self.trainData = np.append(self.trainData, temp)
                self.trainData = self.trainData.reshape([(self.trainData.size / 3), 3])
                self.trainData = self.trainData.astype(np.float32)
            #self.trainData = mapFeature(self.trainData[:,0],self.trainData[:,1],
                                        #self.trainData[:,2]).astype(np.float32)
        if (datatype == 0):
            for f in self.testName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape([(temp.size / 3), 3])
                self.testData = np.append(self.testData, temp)
                self.testData = self.testData.reshape([(self.testData.size / 3), 3])
                self.testData = self.testData.astype(np.float32)
            #self.testData = mapFeature(self.testData[:,0],self.testData[:,1],
                                        #self.testData[:,2]).astype(np.float32)

    def autoSelect(self, K = 30, datatype = 1, bgr2hsv = 1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        flag = cv2.KMEANS_RANDOM_CENTERS
        
        if (datatype == 1):
            for f in self.trainName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape((-1, 3)).astype(np.float32)
                compa_f, label_f, center_f = cv2.kmeans(temp, K, criteria, 10, flag)
                self.trainData = np.vstack((self.trainData, center_f))
            self.trainData = self.trainData.astype(np.float32)
        if (datatype == 0):
            for f in self.testName:
                temp = cv2.imread(f)
                if (bgr2hsv == 1):
                    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
                temp = temp.reshape((-1, 3)).astype(np.float32)
                compa_f,label_f,center_f = cv2.kmeans(temp, K, criteria, 10, flag)
                self.testData = np.vstack((self.testData, center_f))
            self.testData = self.testData.astype(np.float32)
        
            

    def train(self):
        print "\n###training SVM classifier...###\n"
        print "training data shape:", self.trainData.shape
        svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_NU_SVC,
                  nu = 0.1)
        trainLabels = np.repeat([1, 0], (self.trainData.shape[0] / 2))
        trainLabels = trainLabels.reshape([self.trainData.shape[0], 1]).astype(np.float32)
        self.svm.train(self.trainData, trainLabels, params = svm_params)
        print "....!Done!....", self.svm

    def test(self):
        print "\n###testing SVM classifier....###\n"
        print "test data shape:", self.testData.shape
        testLabels = np.repeat([1, 0], (self.testData.shape[0]/2))
        testLabels = testLabels.reshape([self.testData.shape[0], 1]).astype(np.float32)
        result_svm = self.svm.predict_all(self.testData)
        mask = result_svm == testLabels
        correct = np.count_nonzero(mask)
        accuracy_svm = correct * 100.0 / result_svm.size
        print "matches:",correct,"/",self.testData.shape[0]
        print "accuracy:", accuracy_svm

            
        
        
        

        

        

                

        

        
    
    
                
        
        
            
            

    
            
            

        
