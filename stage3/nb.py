from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import PCA, MinMaxScaler
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
assembler = VectorAssembler(inputCols = columns, outputCol = "v_features")
pca = PCA(k = 50, inputCol = assembler.getOutputCol(), outputCol = "p_features")
scaler = MinMaxScaler(inputCol = pca.getOutputCol(), outputCol = "features", min = 0.0, max = 1.0)
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial", featuresCol = "features")
pipeline = Pipeline(stages = [assembler, pca, scaler, nb])

# fit
paramMap = { pca.k: 50 }
model_1 = pipeline.fit(training, paramMap)

paramMap[pca.k] = 2
model_2 = pipeline.fit(training, paramMap)

paramMap[pca.k] = 748
model_3 = pipeline.fit(training, paramMap)

transform(model_1)
transform(model_2)
transform(model_3)

def transform(model):
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
