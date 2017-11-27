import pandas as pd


class DataLoader:

    @staticmethod
    def rank():
        print('start rank data ...')
        feature_type = pd.read_csv('input/features_type.csv')
        feature_type_numeric = feature_type[feature_type.type == 'numeric'].feature

        # 对所有的数值型特征排序,增加模型的稳定性,避免异常值的影响

        # 更合理的做法是将三份数据merge后统一rank
        # 不过三份数据的分布一致,所以此处分别排序也不会有太大影响
        print('rank train_x.')
        x_train = pd.read_csv('input/train_x.csv')
        x_train_rank = pd.DataFrame(x_train.uid, columns=['uid'])
        for feature in feature_type_numeric:
            x_train_rank['r' + feature] = x_train[feature].rank(method='max')
        x_train_rank.to_csv('intermediate_data/train_x_rank.csv', index=None)

        print('rank test_x.')
        x_test = pd.read_csv('input/test_x.csv')
        x_test_rank = pd.DataFrame(x_test.uid, columns=['uid'])
        for feature in feature_type_numeric:
            x_test_rank['r' + feature] = x_test[feature].rank(method='max')
        x_test_rank.to_csv('intermediate_data/test_x_rank.csv', index=None)

        print('rank train_unlabeled.')
        x_train_unlabeled = pd.read_csv('input/train_unlabeled.csv')
        x_train_unlabeled_rank = pd.DataFrame(x_train_unlabeled.uid, columns=['uid'])
        for feature in feature_type_numeric:
            x_train_unlabeled_rank['r' + feature] = x_train_unlabeled[feature].rank(method='max')
        x_train_unlabeled_rank.to_csv('intermediate_data/train_unlabeled_rank.csv', index=None)
        print('rank finish.')
