import os
import json
import numpy as np


class Statistic:
    def __init__(self):
        self.__type_0 = [] # 用于保存 0 类三路自拓扑路径时延协方差数据
        self.__type_1 = [] # 用于保存非 0 类三路自拓扑路径时延协方差数据

    def getWorkspaceDir(self):
        return "/home/zzw/eclipse_workspace/TST/ns-3.30.1/examples/topology-inference/output/statistic"

    def getOutputDir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    def __parseFile(self, file_path):
        with open(file_path, "r") as f:
            content = json.load(f)
            for item in content["data"]:
                temp = []
                temp.append(float(item["covariance_first_second"]))
                temp.append(float(item["covariance_first_third"]))
                temp.append(float(item["covariance_second_third"]))
                if int(item["type"]) == 0:
                    self.__type_0.append(temp)
                else:
                    self.__type_1.append(temp)

    def __save(self):
        file_name_0 = os.path.join(self.getOutputDir(), "0.txt")
        np.savetxt(file_name_0, self.__type_0)
        file_name_1 = os.path.join(self.getOutputDir(), "1.txt")
        np.savetxt(file_name_1, self.__type_1)

    def parseFiles(self):
        dir_name = self.getWorkspaceDir()
        for file_name in os.listdir(dir_name):
            if os.path.splitext(file_name)[1] == ".json":
                self.__parseFile(os.path.join(dir_name, file_name))
        self.__save()
    


if __name__ == "__main__":
    sta = Statistic()
    sta.parseFiles()
