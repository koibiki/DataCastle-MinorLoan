import pandas as pd

print('start rank data ...')
feature_type = pd.read_csv('../input/features_type.csv')
feature_type_numeric = feature_type[feature_type.type == 'numeric'].feature

#对所有的数值型特征排序,增加模型的稳定性,避免异常值的影响

#更合理的做法是将三份数据merge后统一rank
#不过三份数据的分布一致,所以此处分别排序也不会有太大影响
X_train = pd.read_csv('../input/train_x.csv')
X_train_rank = pd.DataFrame(X_train.uid, columns=['uid'])
for feature in feature_type_numeric:
    X_train_rank['r' + feature] = X_train[feature].rank(method='max')
X_train_rank.to_csv('../intermediate_data/train_x_rank.csv', index=None)

X_test = pd.read_csv('../input/test_x.csv')
X_test_rank = pd.DataFrame(X_test.uid, columns=['uid'])
for feature in feature_type_numeric:
    X_test_rank['r' + feature] = X_test[feature].rank(method='max')
X_test_rank.to_csv('../intermediate_data/test_x_rank.csv', index=None)

X_train_unlabeled_1 = pd.read_csv('../input/train_unlabeled_1.csv')
X_train_unlabeled_2 = pd.read_csv('../input/train_unlabeled_2.csv')
X_train_unlabeled_3 = pd.read_csv('../input/train_unlabeled_3.csv')

X_train_unlabeled = pd.concat([X_train_unlabeled_1, X_train_unlabeled_2, X_train_unlabeled_3], axis=0)
X_train_unlabeled_rank = pd.DataFrame(X_train_unlabeled.uid, columns=['uid'])
for feature in feature_type_numeric:
    X_train_unlabeled_rank['r' + feature] = X_train_unlabeled[feature].rank(method='max')
X_train_unlabeled_rank.to_csv('../intermediate_data/train_unlabeled_rank.csv', index=None)
print('rank finish.')