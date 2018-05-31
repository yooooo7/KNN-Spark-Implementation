from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--treenums", help = "PCA dimension", default = 50)
args = parse.parse_args()
tree_nums = int(args.treenums)

spark = SparkSession \
    .builder \
    .appName("Spark RF implementation") \
    .getOrCreate()

DATA_PATH = "/share/demo/MNIST-sample/"
test_file = 'Test-1000.csv'
train_file = 'Train-6000.csv'

# prepare training&test file
training = spark.read.csv(DATA_PATH + train_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')
test = spark.read.csv(DATA_PATH + test_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')

# Configure an ML pipeline, which consists of two stages: assembler, pca, naive bayes
columns = training.columns[1:]
assembler = VectorAssembler(inputCols = columns, outputCol = "features")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
pipeline = Pipeline(stages = [assembler, rf])

# fit
paramMap = { rf.numTrees: tree_nums }
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
