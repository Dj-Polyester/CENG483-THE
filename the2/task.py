import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
from pprint import PrettyPrinter
pp = PrettyPrinter()


TRAINOBJSIZE = 1500
VALIDATIONFOLDER = "validation"


class DenseSift():
    def __init__(self, gridSize, stepSize=None, offset: np.array = None) -> None:
        self.gridSize = gridSize
        if stepSize == None:
            self.stepSize = gridSize
        if offset == None:
            self.offset = np.array([gridSize/2, gridSize/2])

    def detect(self, img):
        M, N = int(np.ceil(img.shape[0]/self.stepSize)
                   ), int(np.ceil(img.shape[1]/self.stepSize))
        x, y = np.meshgrid(np.arange(N), np.arange(M))
        xy = np.stack((x, y), axis=2)
        xy = self.stepSize*xy+self.offset
        xy = xy.reshape((M*N, 2))
        return cv2.KeyPoint_convert(points2f=xy.tolist(), size=self.gridSize)


def filePaths(path):
    return [os.path.join(root, fileName) for root, _, files in os.walk(path) for fileName in files]


def getDists(a, b, M, N):
    a3d = np.stack((a,)*N, axis=1)
    b3d = np.stack((b,)*M, axis=0)
    return np.linalg.norm(b3d-a3d, axis=2)


class The2():
    def __init__(self, the2DataPath) -> None:
        self.the2DataPath = the2DataPath

        trainPath = f"{self.the2DataPath}/train"
        validationPath = f"{self.the2DataPath}/{VALIDATIONFOLDER}"

        self.trainObjPaths = np.random.choice(
            filePaths(trainPath), TRAINOBJSIZE, replace=False)
        self.validationObjPaths = filePaths(validationPath)
        self.validationObjNames = [os.path.basename(
            qfname) for qfname in self.validationObjPaths]

        self.descriptions = None
        self.words = None
        self.results = None

        self.classNames = np.array(os.listdir(trainPath))
        classIndices = list(range(self.classNames.size))
        self.classdic = dict(zip(self.classNames, classIndices))

    def configure(self, detMethod, **kwargs):
        self.detectionMethod = detMethod
        if self.detectionMethod == "interest":
            self.kwargs = kwargs
            self.sift = cv2.SIFT_create(**kwargs)
            self.fast = cv2.FastFeatureDetector_create()
            self.detect = self.fast.detect
        elif self.detectionMethod == "dense":
            self.kwargs = kwargs
            self.sift = cv2.SIFT_create()
            self.dense = DenseSift(**kwargs)
            self.detect = self.dense.detect

    def pipeline(self, kmeansk=128, knnk=8, load=False):
        self.detectAndDescribeAll()
        self.kmeansCluster(kmeansk)
        if VALIDATIONFOLDER == "validation":
            return self.accuracy(knnk, load)
        self.test(knnk, load)

    # DESCRIPTIONS

    def detectAndDescribeAll(self):
        tmp = [self.detectAndDescribeFromFile(
            fname) for fname in self.trainObjPaths]
        self.descriptions = np.concatenate(tmp)
        np.save(self.detectionMethod, self.descriptions)

    def detectAndDescribeFromFile(self, fname):
        img = cv2.imread(fname)
        return self.detectAndDescribe(img)[1]

    def detectAndDescribe(self, img):
        kps = self.detect(img)
        return self.sift.compute(img, kps)
    # WORDS

    def kmeansCluster(self, k=128) -> np.array:
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(self.descriptions)
        self.words = kmeans.cluster_centers_
        np.save(f"{self.detectionMethod}_words", self.words)

    # REPRESENT IMG
    def hist(self, img):
        _, des = self.detectAndDescribe(img)

        M, N = des.shape[0], self.words.shape[0]
        dists = getDists(des, self.words, M, N)

        closestWords = np.argmin(dists, axis=1)
        return np.histogram(closestWords, bins=np.arange(N+1))

    def histFromFile(self, fname):
        img = cv2.imread(fname)
        return self.hist(img)[0]
    # CLASSIFY

    def test(self, k=8, load=False):
        if load:
            self.words = np.load(f"{self.detectionMethod}_words.npy")
        self.acquireResults(k)

    def accuracy(self, k=8, load=False):
        if load:
            self.words = np.load(f"{self.detectionMethod}_words.npy")

        self.actual = np.array([self.classdic[os.path.basename(os.path.dirname(qfname))]
                                for qfname in self.validationObjPaths])
        M = self.acquireResults(k)

        return np.sum(self.actual == self.results)/M

    def acquireResults(self, k):
        tlbls = np.array([self.classdic[os.path.basename(os.path.dirname(tfname))]
                          for tfname in self.trainObjPaths])

        qhists = np.stack([self.histFromFile(qfname)
                           for qfname in self.validationObjPaths])
        thists = np.stack([self.histFromFile(tfname)
                           for tfname in self.trainObjPaths])

        M, N = qhists.shape[0], thists.shape[0]
        dists = getDists(qhists, thists, M, N)
        kclosest = np.argsort(dists, axis=1)[:, :k]

        kclosest1d = kclosest.reshape(k*M)
        tclosest = tlbls[kclosest1d].reshape(M, k)
        self.results = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 1, tclosest)
        return M

    def confusionMatrix(self):
        M = self.classNames.size
        self.confusion, _, __ = np.histogram2d(
            self.actual, self.results, bins=(np.arange(M+1), np.arange(M+1)))
        pp.pprint(self.confusion.astype(int).tolist())

    def print(self):
        filename = "_".join([f"{k}_{v}" for k, v in self.kwargs.items()])
        resultsLiteral = self.classNames[self.results]
        with open(f"{filename}.txt", "w") as f:
            f.writelines([f"{objname}: {result}\n" for objname,
                          result in zip(self.validationObjNames, resultsLiteral)])


if __name__ == "__main__":
    # nfeatures = 0
    # nOctaveLayers = 3,
    # contrastThreshold = 0.04,
    # edgeThreshold = 10,
    # sigma = 1.6
    the2 = The2("the2_data")
    if VALIDATIONFOLDER == "validation":
        print("1) SIFT - nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, kmeansk = 128, knnk = 8")
        listOfArgs = [{"nfeatures": 80},
                      {"nOctaveLayers": 10},
                      {"contrastThreshold": 0.08},
                      {"edgeThreshold": 5},
                      {"sigma": 0.50}, ]

        the2.configure("interest")
        print(f"Default: {the2.pipeline()}")
        for kwargs in listOfArgs:
            the2.configure("interest", **kwargs)
            print(
                f"{list(kwargs.keys())[0]} = {list(kwargs.values())[0]}: {the2.pipeline()}")
            # the2.print()

        gridSizes = [4, 8, 16]
        print("1) dSIFT - grid size, kmeansk = 128, knnk = 8")
        for i in gridSizes:
            the2.configure("dense", gridSize=i)
            print(f"{i}: {the2.pipeline()}")
            # the2.print()

        print("2) BoW - kmeansk, sigma = 0.5")
        kmeansk = [32, 64, 128]
        for k in kmeansk:
            the2.configure("interest", sigma=0.5)
            print(
                f"{k}: {the2.pipeline(kmeansk=k)}")
            # the2.print()
        print("3) Classification - knnk, sigma = 0.5, kmeansk = 128")
        knnk = [16, 32, 64]
        for k in knnk:
            the2.configure("interest", sigma=0.5)
            print(
                f"{k}: {the2.pipeline(kmeansk=128, knnk=k)}")
            the2.confusionMatrix()
            # the2.print()
    else:
        # Optimum params
        the2.configure("interest", sigma=0.5)
        the2.pipeline(kmeansk=128, knnk=32)
        the2.print()
