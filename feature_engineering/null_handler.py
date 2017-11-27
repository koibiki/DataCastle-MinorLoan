import pandas as pd


class NullHandler:

    def null_count(self):
        print('calculate null feature count.')
        self.__classifier(self, 'input/train_x.csv', 'intermediate_data/train_x_null.csv')
        self.__classifier(self, 'input/test_x.csv', 'intermediate_data/test_x_null.csv')
        self.__classifier(self, 'input/train_unlabeled.csv', 'intermediate_data/train_unlabeled_null.csv')
        print('write null feature data.')

    @staticmethod
    def __classifier(self, data_path, output_path):
        data = pd.read_csv(data_path)
        data['null_count'] = (data < 0).sum(axis=1)
        data['null_classify'] = data['null_count'].apply(self.__classify)
        data[['uid', 'null_count', 'null_classify']].to_csv(output_path)

    @staticmethod
    def __classify(null_count):
        if null_count <= 32:
            return 1
        elif 32 < null_count <= 69:
            return 2
        elif 69 < null_count <= 147:
            return 3
        elif 147 < null_count <= 194:
            return 4
        else:
            return 5
