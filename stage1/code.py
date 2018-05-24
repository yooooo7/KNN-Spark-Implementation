from units import *

def main():
    # DATA_PATH = "/share/MNIST/"
    DATA_PATH = ''
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

    # format result and output
    formatted_res = f_res(result)
    print(formatted_res.take(20))
    # result.saveAsTextFile(output_path)

    # show precision, recall and f1-score overall
    show_metrics(result)

    stop_context()

if __name__ == "__main__":
    main()
