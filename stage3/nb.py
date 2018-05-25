from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

spark = SparkSession \
    .builder \
    .appName("Spark NB implementation") \
    .getOrCreate()

DATA_PATH = "/share/MNIST/"
test_file = 'Test-label-28x28.csv'
train_file = 'Train-label-28x28.csv'

# prepare training&test file
training = spark.read.csv(DATA_PATH + train_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')
test = spark.read.csv(DATA_PATH + test_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')

# Configure an ML pipeline, which consists of two stages: assembler, pca, naive bayes
columns = training.columns[1:]
assembler = VectorAssembler(inputCols = columns, outputCol = "features")
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial")
pipeline = Pipeline(stages = [assembler, nb])

# fit
model = pipeline.fit(training)

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