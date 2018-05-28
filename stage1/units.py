from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.accumulators import AccumulatorParam
import numpy as np
import argparse

spark = SparkSession \
    .builder \
    .appName("Spark KNN implementation") \
    .getOrCreate()

LABEL_NUM = 10

class ListParam(AccumulatorParam):
    def zero(self, v):
        return [0] * len(v)
    def addInPlace(self, acc1, acc2):
        for i in range(len(acc1)):
            acc1[i] += acc2[i]
        return acc1

TP_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())
FP_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())
FN_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())

parse = argparse.ArgumentParser()
parse.add_argument("--dimension", help = "PCA dimension", default = 50)
parse.add_argument("--k", help = "k nearest", default = 10)
args = parse.parse_args()
dimension = int(args.dimension)
k = int(args.k)

REPRETATION_NUM = 16

def read_CSV(DATA_PATH, datafile):
    test_df = spark.read.csv(DATA_PATH + datafile, header = False, inferSchema = "true")
    return test_df

def vectorization(df):
    assembler = VectorAssembler(
        inputCols = df.columns[1:],
        outputCol = "features"
        )
    test_vectors = assembler.transform(df).select(df["_c0"].alias("label"),"features")
    return test_vectors

class pca_m(object):
    def __init__(self, dimension):
        self.pca_model = PCA(k = dimension, inputCol = "features", outputCol = 'pca')

    def fit_train(self, train_vectors):
        self.test_fitted = self.pca_model.fit(train_vectors)

    def pca_process(self, vectors):
        pca_result = self.test_fitted.transform(vectors).select("label","pca")
        return pca_result

def divide_train(train_pca):
    tr_pca = np.array(train_pca.select(train_pca['pca']).collect())[:, 0, :]
    tr_label = np.array(train_pca.select(train_pca['label']).collect())
    return tr_pca, tr_label

tr_data = []
tr_l = []
class KNN(object):
    def __init__(self, tr_pca, tr_label):
        global tr_data
        global tr_l
        tr_data = spark.sparkContext.broadcast(tr_pca)
        tr_l = spark.sparkContext.broadcast(tr_label)

    @staticmethod
    def getNeighbours(record):
        label, test_features = record
        test_features = np.array(test_features)
        train = tr_data.value
        tr_label = tr_l.value
        dis = np.sqrt( np.sum( ((train - test_features) ** 2), axis = 1 ))[:, np.newaxis]
        com = np.concatenate((tr_label, dis), axis = 1)
        sorted_com = sorted(com, key = lambda x: x[1])[:k]
        tk_label = [x[0] for x in sorted_com]
        counts = np.bincount(tk_label)
        prediction = np.argmax(counts).item()

        prediction = int(prediction)
        label = int(label)
        global TP_counter, FP_counter, FN_counter
        c = [0] * LABEL_NUM
        if prediction == label:
            c[prediction] = 1
            TP_counter += c
            c[prediction] = 0
        else:
            c[label] = 1
            FN_counter += c
            c[label] = 0

            c[prediction] = 1
            FP_counter += c
            c[prediction] = 0

        return ("label: {}, prediction: {};".format(label, prediction))

    def predict(self, test_pca):
        self.result = test_pca.rdd.map(self.getNeighbours)
        return self.result

    def show_metrics(self):

        for i in range(LABEL_NUM):
            TPs = TP_counter.value
            FPs = FP_counter.value
            FNs = FN_counter.value
            label = str(i)
            p = round(TPs[i] / float( TPs[i] + FPs[i] ), 2)
            r = round(TPs[i] / float( TPs[i] + FNs[i] ), 2)
            F1_score = round(2*p*r / (p + r), 2)
            print("label: {}\nprecision: {}\nrecall: {}\nf1-score: {}\n".format(label, p, r, F1_score))

def stop_context():
    spark.stop()

def main():
    DATA_PATH = "/share/MNIST/"
    test_file = 'Test-label-28x28.csv'
    train_file = 'Train-label-28x28.csv'

    pca = pca_m(dimension)

    # pre process for train dataset
    train_df = read_CSV(DATA_PATH, train_file)
    train_vectors = vectorization(train_df)
    pca.fit_train(train_vectors)
    train_pca = pca.pca_process(train_vectors)

    # pre process for test dataset
    test_df = read_CSV(DATA_PATH, test_file).repartition(REPRETATION_NUM)
    test_vectors = vectorization(test_df)
    test_pca = pca.pca_process(test_vectors)

    # divide train data to features and labels
    tr_pca, tr_label = divide_train(train_pca)

    # KNN
    knn_m = KNN(tr_pca, tr_label)
    result = knn_m.predict(test_pca)

    # persist result after next action for future calculate
    result.persist()

    # save result
    result.collect()

    # for each label, show precision recall and f1-score
    knn_m.show_metrics()

    stop_context()

if __name__ == "__main__":
    main()
