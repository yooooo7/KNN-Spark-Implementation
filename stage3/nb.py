from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import PCA, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--dimension", help = "PCA dimension", default = 50)
args = parse.parse_args()
dimension = int(args.dimension)

spark = SparkSession \
    .builder \
    .appName("Spark NB implementation") \
    .getOrCreate()

DATA_PATH = "/share/demo/MNIST-sample/"
test_file = 'Test-1000.csv'
train_file = 'Train-6000.csv'

# prepare training&test file
training = spark.read.csv(DATA_PATH + train_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')
test = spark.read.csv(DATA_PATH + test_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')

# Configure an ML pipeline, which consists of four stages: assembler, pca, scaler, naive bayes
columns = training.columns[1:]
assembler = VectorAssembler(inputCols = columns, outputCol = "v_features")
pca = PCA(k = 50, inputCol = assembler.getOutputCol(), outputCol = "p_features")
scaler = MinMaxScaler(inputCol = pca.getOutputCol(), outputCol = "features", min = 0.0, max = 1.0)
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial", featuresCol = "features")
pipeline = Pipeline(stages = [assembler, pca, scaler, nb])

# fit
paramMap = { pca.k: dimension }
model = pipeline.fit(training, paramMap)

# predict
prediction = model.transform(test)
prediction = prediction.select(['label', 'prediction'])

prediction.show(5)

# metrics
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)

print("Test set accuracy =" + str(accuracy))

# stop env
spark.stop()
