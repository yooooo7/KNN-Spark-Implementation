from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA

spark = SparkSession \
    .builder \
    .appName("Spark RF&NB implementation") \
    .getOrCreate()

DATA_PATH = "/share/MNIST/"
# DATA_PATH = ''
test_file = 'Test-label-28x28.csv'
train_file = 'Train-label-28x28.csv'

# prepare training file
training = spark.read.csv(DATA_PATH + train_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')

# Configure an ML pipeline, which consists of three stages: assembler, pca, naive bayes
columns = training.columns[1:]
assembler = VectorAssembler(inputCols = columns, outputCol = "features")
# pca = PCA(k = 50, inputCol = assembler.getOutputCol(), outputCol = 'features')
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial")
pipeline = Pipeline(stages = [assembler, nb])

model = pipeline.fit(training)

# prepare test file
test = spark.read.csv(DATA_PATH + test_file, header = False, inferSchema = "true").withColumnRenamed("_c0", 'label')

prediction = model.transform(test)
prediction = prediction.select(['label','features', 'probability', 'prediction'])

prediction.show(5)
