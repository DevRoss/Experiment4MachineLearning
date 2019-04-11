# coding:utf-8

from sklearn.model_selection import StratifiedKFold
import numpy as np


class DataModel:
    def __init__(self):
        self.files = ['data/0_cut.txt', 'data/1_cut.txt']
        self.sentences, self.labels = self._get_load_data()

    def _get_load_data(self):
        sentences = []
        label_ = []
        for i, file in enumerate(self.files):
            with open(file, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                sentences.extend(map(lambda x: x.strip().split(), lines))
                label_.extend([i] * len(lines))
        return np.array(sentences), np.array(label_)

    def get_fold(self, num_fold=3):
        skf = StratifiedKFold(num_fold, random_state=5)
        for train_index, test_index in skf.split(self.sentences, self.labels):
            X_train, X_test = self.sentences[train_index], self.sentences[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            yield X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_model = DataModel()

