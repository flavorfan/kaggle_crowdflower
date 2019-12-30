import sys
import pickle
import numpy as np
import pandas as pd
import joblib


from Feat import nlp_utils
# sys.path.append("/home/algo/code/gitrepo/pylib/Kaggle_CrowdFlower/Code")



if __name__ == '__main__':
    ###############
    ## Load Data ##
    ###############
    print("Load data...")
    from param_config import ParamConfig
    config = ParamConfig(feat_folder="./feat_folder",
                         drop_html_flag=True,
                         stemmer_type="porter",
                         cooccurrence_word_exclude_stopword=False)

    dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
    dfTest = pd.read_csv(config.original_test_data_path).fillna("")
    # number of train/test samples
    num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

    print("Done.")

    ######################
    ## Pre-process Data ##
    ######################
    print("Pre-process data...")

    ## insert fake label for test
    dfTest["median_relevance"] = np.ones((num_test))
    dfTest["relevance_variance"] = np.zeros((num_test))

    ## insert sample index
    dfTrain["index"] = np.arange(num_train)
    dfTest["index"] = np.arange(num_test)

    ## one-hot encode the median_relevance
    for i in range(config.n_classes):
        dfTrain["median_relevance_%d" % (i + 1)] = 0
        dfTrain["median_relevance_%d" % (i + 1)][dfTrain["median_relevance"] == (i + 1)] = 1

    ## query ids
    qid_dict = dict()
    for i, q in enumerate(np.unique(dfTrain["query"]), start=1):
        qid_dict[q] = i

    ## insert query id
    dfTrain["qid"] = map(lambda q: qid_dict[q], dfTrain["query"])
    dfTest["qid"] = map(lambda q: qid_dict[q], dfTest["query"])

    ## clean text
    # pickle 模块不能序列化lambda，使用自定义函数
    def clean(line):
        nlp_utils.clean_text(line, drop_html_flag=config.drop_html_flag)

    # clean = lambda line: nlp_utils.clean_text(line, drop_html_flag=config.drop_html_flag)
    dfTrain = dfTrain.apply(clean, axis=1)
    dfTest = dfTest.apply(clean, axis=1)

    print("Done.")

    ###############
    ## Save Data ##
    ###############
    print("Save data...")


    with open(config.processed_train_data_path, "wb") as f:
        # pickle.dump(dfTrain, f, -1)
        pickle.dump(dfTrain, f)

    with open(config.processed_test_data_path, "wb") as f:
        # pickle.dump(dfTest, f, -1)
        pickle.dump(dfTest, f)

    # joblib.dump(dfTrain,config.processed_train_data_path )
    # joblib.dump(dfTest,config.processed_test_data_path)
    print("Done.")

