from utils import *

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

    # save result
    result.collect()

    # for each label, show precision recall and f1-score
    knn_m.show_metrics()

    # stop spark session instance
    stop_context()

if __name__ == "__main__":
    main()
