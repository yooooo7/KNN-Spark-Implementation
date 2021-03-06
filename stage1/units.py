from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.accumulators import AccumulatorParam
from pyspark.ml import Pipeline
import numpy as np
import argparse

spark = SparkSession \
    .builder \
    .appName("Spark KNN implementation") \
    .getOrCreate()

DATA_PATH = "/share/demo/MNIST-sample/"
test_file = 'Test-1000.csv'
train_file = 'Train-6000.csv'

LABEL_NUM = 10
REPRETATION_NUM = 16

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
parse.add_argument("--dimension", help = "PCA dimension")
parse.add_argument("--k", help = "k nearest")
parse.add_argument("--totalcore", help = "total core")
args = parse.parse_args()
dimension = int(args.dimension)
k = int(args.k)
total_core = int(args.totalcore)

REPRETATION_NUM = total_core*2

test_df = spark.read.csv(DATA_PATH + test_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')
train_df = spark.read.csv(DATA_PATH + train_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')
columns = train_df.columns[1:]
assembler = VectorAssembler(inputCols = columns, outputCol = "features")
pca = PCA(k = 50, inputCol = "features", outputCol = 'pca')

pipeline = Pipeline(stages = [assembler, pca])

# fit PCA with train data
paramMap = { pca.k: dimension }
model = pipeline.fit(train_df, paramMap)

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
        # get broadcast value
        train = tr_data.value
        tr_label = tr_l.value
        # Caculate Euclidean distance
        dis = np.sqrt( np.sum( ((train - test_features) ** 2), axis = 1 ))
        ids = np.argpartition(dis, k)[:k]
        # get k nearest neighbours
        nearest_dists = np.take(tr_label, ids)
        # vote
        counts = np.bincount(nearest_dists)
        prediction = np.argmax(counts).item()

        # accumulate TP, FP and FN
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

        return (label, prediction)

    def predict(self, test_pca):
        self.result = test_pca.rdd.map(self.getNeighbours)
        return self.result

    def show_metrics(self):
        # for each label, print precision, recall and f1-score
        for i in range(LABEL_NUM):
            TPs = TP_counter.value
            FPs = FP_counter.value
            FNs = FN_counter.value
            label = str(i)
            p = round(TPs[i] / float( TPs[i] + FPs[i] ), 3)
            r = round(TPs[i] / float( TPs[i] + FNs[i] ), 3)
            F1_score = round(2*p*r / (p + r), 3)
            print("label: {}\tprecision: {}  recall: {}  f1-score: {}\n".format(label, p, r, F1_score))

def stop_context():
    spark.stop()

def main():
    train_pca = model.transform(train_df).select(['label', 'pca'])
    test_pca =  model.transform(test_df).select(['label', 'pca']).repartition(REPRETATION_NUM)

    # divide train data to features and labels
    tr_pca, tr_label = divide_train(train_pca)

    # KNN
    knn_m = KNN(tr_pca, tr_label)
    result = knn_m.predict(test_pca)

    # persist result after next action for future calculate
    result.persist()

    # collect result
    print(result.collect())

    # for each label, show precision recall and f1-score
    knn_m.show_metrics()

    # stop spark session instance
    stop_context()

if __name__ == "__main__":
    main()
