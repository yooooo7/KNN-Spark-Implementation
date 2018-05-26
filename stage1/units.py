from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.sql.types import StructField, FloatType, StructType, StringType
import numpy as np
import argparse
from pyspark.accumulators import AccumulatorParam

spark = SparkSession \
    .builder \
    .appName("Spark KNN implementation") \
    .getOrCreate()

LABEL_NUM = 10

class ListParam(AccumulatorParam):
    def zero(self, v):
        return [0] * len(v)
    def addInPlace(self, acc1, acc2):
        for i in xrange(len(acc1)):
            acc1[i] += acc2[i]
        return acc1

TP_counter = spark.sparkContext.accumulator([0 for i in range(LABEL_NUM)], ListParam())
FP_counter = spark.sparkContext.accumulator([0 for i in range(LABEL_NUM)], ListParam())
FN_counter = spark.sparkContext.accumulator([0 for i in range(LABEL_NUM)], ListParam())

def init_par():
    parse = argparse.ArgumentParser()
    parse.add_argument("--dimension", help = "PCA dimension", default = 50)
    parse.add_argument("--k", help = "k nearest", default = 5)
    parse.add_argument("--output", help="the output path", default="as2_stage1/outp_5.25")
    args = parse.parse_args()
    dimension = int(args.dimension)
    k = int(args.k)
    outp_path = args.output
    return (dimension, k, outp_path)

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

    def fit_test(self, test_vectors):
        self.test_fitted = self.pca_model.fit(test_vectors)

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
        sorted_com = sorted(com, key = lambda x: x[1])[:5]
        tk_label = [x[0] for x in sorted_com]
        counts = np.bincount(tk_label)
        res = np.argmax(counts).item()
        return (int(res), int(label))

    @staticmethod
    def conf_matrix(record):
        global TP_counter
        global FP_counter
        global FN_counter
        prediction, label = record
        print(prediction)
        print(label)
        if prediction == label:
            TP_counter[prediction] += 1
        else:
            FN_counter[label] += 1
            FP_counter[prediction] += 1

    def predict(self, test_pca):
        self.result = test_pca.rdd.map(self.getNeighbours)
        return self.result

    def con_m(self):
        self.result.foreach(conf_matrix)

        global TP_counter
        global FP_counter
        global FN_counter

        print(TP_counter)
        print(FP_counter)
        print(FN_counter)

def stop_context():
    spark.stop()
