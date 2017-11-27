import pandas as pd


class Discretizator:

    """对数值型特征进行分桶"""

    def data_discretization(self):
        self.discret('intermediate_data/train_x_rank.csv', 'intermediate_data/train_x_discretization.csv')

    @staticmethod
    def discret(data_path, output_path):
        features_type = pd.read_csv('input/features_type.csv')
        features_type_numeric = features_type[features_type.type == 'numeric'].feature
        features_type_category = features_type[features_type.type == 'category'].feature
        data = pd.read_csv(data_path)
        rank_count = data.shape[0]

        data_uid = data['uid']
        data_numeric = data[features_type_numeric]
        data_category = data[features_type_category]

        data_numeric[data_numeric <= rank_count / 10] = 1
        data_numeric[rank_count / 10 < data_numeric <= rank_count / 5] = 2
        data_numeric[rank_count / 5 < data_numeric <= rank_count * 3 / 10] = 3
        data_numeric[rank_count * 3 / 10 < data_numeric <= rank_count * 2 / 5] = 4
        data_numeric[rank_count * 2 / 5 < data_numeric <= rank_count * 1 / 2] = 5
        data_numeric[rank_count * 1 / 2 < data_numeric <= rank_count * 3 / 5] = 6
        data_numeric[rank_count * 3 / 5 < data_numeric <= rank_count * 7 / 10] = 7
        data_numeric[rank_count * 7 / 10 < data_numeric <= rank_count * 4 / 5] = 8
        data_numeric[rank_count * 4 / 5 < data_numeric <= rank_count * 9 / 10] = 9
        data_numeric[rank_count * 9 / 10 < data_numeric] = 10
        pd.concat([data_uid, data_numeric, data_category], axis=1).to_csv(output_path)

