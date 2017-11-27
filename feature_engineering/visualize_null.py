import pandas as pd
import matplotlib.pyplot as plot
import os


class Visualize:

    def visualize_null(self):
        train_data_path = 'intermediate_data/train_x_null.csv'
        test_data_path = 'intermediate_data/test_x_null.csv'
        train_unlabeled_data_path = 'intermediate_data/train_unlabeled_null.csv'
        data_paths = [train_data_path, test_data_path, train_unlabeled_data_path]
        for data_path in data_paths:
            if os.path.isfile(train_data_path):
                self.__visualize_data(data_path)

    @staticmethod
    def __visualize_data(data_path):
        data = pd.read_csv(data_path)
        data = data.sort_values('null_count')
        data = data.reset_index()
        plot.xlim(-0.1 * data.shape[0], data.shape[0] * 1.1)
        plot.ylim(-500, 2000)
        plot.xlabel('uid', fontsize=18, labelpad=5)
        plot.ylabel('absent feature count', fontsize=18, labelpad=5)
        x = data.index
        y = data.null_count.values
        plot.scatter(x, y)
        plot.show()
