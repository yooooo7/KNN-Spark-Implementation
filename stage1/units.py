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
        for i in range(len(acc1)):
            acc1[i] += acc2[i]
        return acc1

TP_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())
FP_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())
FN_counter = spark.sparkContext.accumulator([0] * LABEL_NUM, ListParam())

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
        global TP_counter, FP_counter, FN_counter
        prediction, label = record
        m = [0] * LABEL_NUM
        n1 = [0] * LABEL_NUM
        n2 = [0] * LABEL_NUM
        if prediction == label:
            m[prediction] = 1
            TP_counter += m
        else:
            n1[label] = 1
            FN_counter += n1
            n2[prediction] = 1
            FP_counter += n2

    def predict(self, test_pca):
        self.result = test_pca.rdd.map(self.getNeighbours)
        return self.result

    def con_m(self):
        self.result.foreach(self.conf_matrix)

        for i in range(LABEL_NUM):
            TPs = TP_counter.value
            FPs = FP_counter.value
            FNs = FN_counter.value
            label = str(i)
            p = TPs[i] / float( TPs[i] + FPs[i] )
            r = TPs[i] / float( TPs[i] + FNs[i] )
            F1_score = 2*p*r / (p + r)
            print("label: {}\nprecision: {}\nrecall: {}\nf1-score: {}\n".format(label, p, r, F1_score))


def stop_context():
    spark.stop()

def main():
    DATA_PATH = "/share/MNIST/"
    test_file = 'Test-label-28x28.csv'
    train_file = 'Train-label-28x28.csv'

    dimension, k, output_path = init_par()
    pca = pca_m(dimension)

    # pre process for test dataset
    test_df = read_CSV(DATA_PATH, test_file)
    test_vectors = vectorization(test_df)
    pca.fit_test(test_vectors)
    test_pca = pca.pca_process(test_vectors)

    # pre process for train dataset
    train_df = read_CSV(DATA_PATH, train_file)
    train_vectors = vectorization(train_df)
    train_pca = pca.pca_process(train_vectors)

    # divide train data to features and labels
    tr_pca, tr_label = divide_train(train_pca)

    # KNN
    knn_m = KNN(tr_pca, tr_label)
    result = knn_m.predict(test_pca)
    print(result.take(5))

    knn_m.con_m()

    # format result and output
    def format(record):
        pre, ori = record
        return ("label: {}, prediction: {};".format(ori, pre))

    # formatted_res = result.map(format)
    # formatted_res.saveAsTextFile(output_path)

    stop_context()

if __name__ == "__main__":
    main()
