"""
该类用于生成正太分布的数据，将生成的数据保存，或者从保存数据的文件将数据读出
"""

import numpy as np
import os
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.__m_train_path = self.get_train_path()
        self.__m_test_path = self.get_test_path()
        self.__m_test_label_path = self.get_test_label_path()
        self.__m_train_data = None
        self.__m_test_data = None
        self.__m_test_label = None
        self.__max_number = None

    def data_generator(self,
                       type_0_mean,
                       type_0_sigma,
                       type_1_mean,
                       type_1_sigma,
                       train_size,
                       test_size):
        train_data = []
        test_data = []
        test_label = []

        for i in range(train_size):
            train_data.append(round(np.random.normal(type_1_mean, type_1_sigma), 2))

        for _ in range(int(0.5*test_size)):
            test_data.append(round(np.random.normal(type_1_mean, type_1_sigma), 2))
            test_label.append(1)

        for _ in range(test_size - int(0.5*test_size)):
            test_data.append(round(np.random.normal(type_0_mean, type_0_sigma), 2))
            test_label.append(0)

        np.savetxt(self.__m_train_path, train_data, fmt="%.2f")
        np.savetxt(self.__m_test_path, test_data, fmt="%.2f")
        np.savetxt(self.__m_test_label_path, test_label, fmt="%d")

    def load_data(self):
        self.__m_train_data = np.loadtxt(self.__m_train_path)
        self.__m_test_data = np.loadtxt(self.__m_test_path)
        self.__m_test_label = np.loadtxt(self.__m_test_label_path)

    def normalize(self):
        self.__max_number = max(self.__m_train_data)
        print("最大值：", self.__max_number)
        self.__m_train_data = np.divide(self.__m_train_data, self.__max_number)

    @staticmethod
    def plot(data_list: list):
        data_list.sort()
        plt.hist(data_list, bins=40)
        plt.show()

    @staticmethod
    def get_dir():
        return os.path.dirname(__file__)

    def get_data_dir(self):
        path = os.path.join(self.get_dir(), "data_sets")
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_train_path(self):
        return os.path.join(self.get_data_dir(), "train.txt")

    def get_test_path(self):
        return os.path.join(self.get_data_dir(), "test.txt")

    def get_test_label_path(self):
        return os.path.join(self.get_data_dir(), "test_label.txt")

    @property
    def train_data(self):
        return self.__m_train_data

    @property
    def max_number(self):
        return self.__max_number


if __name__ == '__main__':
    data = Data()
    # data.data_generator(5,
    #                     2,
    #                     20,
    #                     10,
    #                     20000,
    #                     1000)
    data.load_data()
    # data.normalize()
    data.plot(list(data.train_data))
