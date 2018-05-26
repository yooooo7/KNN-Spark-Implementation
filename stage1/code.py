from units import *

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

    knn_m.show_metrics()

    # format result and output
    def format(record):
        pre, ori = record
        return ("label: {}, prediction: {};".format(ori, pre))

    # formatted_res = result.map(format)
    # formatted_res.saveAsTextFile(output_path)

    stop_context()

if __name__ == "__main__":
    main()
