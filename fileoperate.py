import json
import numpy as np
import matplotlib.pyplot as plt
import os

#计算平均相对差异
def mean_diff(covlist):
    cov12=covlist[0]
    cov13=covlist[1]
    cov23=covlist[2]
    mean_cov=(1/3)*(cov12+cov13+cov23)
    mean_diff_result=(1/3)*(abs(cov12-mean_cov)+abs(cov13-mean_cov)+abs(cov23-mean_cov))/mean_cov
    # print(mean_cov)
    # print(mean_diff_result)
    return mean_diff_result

#读取json的文件
def fileread():
    filelist=os.listdir('type')
    all_cov_data = []
    for a_json_file in filelist:
        with open('type/'+a_json_file, 'r')as fr:
            print(a_json_file)
            python_data=json.load(fr)
            print(python_data)
            #print(python_data['data'])
            for item in python_data['data']:
                a_cov_data=[]
                a_cov_data.append(item['covariance_first_second'])
                a_cov_data.append(item['covariance_first_third'])
                a_cov_data.append(item['covariance_second_third'])
                all_cov_data.append(a_cov_data)
    all_mean_diff=[]
    for item in all_cov_data:
        a_mean_diff=mean_diff(item)
        all_mean_diff.append(a_mean_diff)
    print(all_mean_diff)

#读取txt的文件
def txtFileRead():
    all_cov_data=[]
    all_cov_data_0=[]
    all_cov_data_1=[]
    with open("output/0.txt",'r') as fr:
        while True:
            oneline=fr.readline()
            if oneline:
                oneline=oneline.rstrip()
                a_cov_data=oneline.split(' ')
                for item in range(len(a_cov_data)):
                    a_cov_data[item]=float(a_cov_data[item])
                # print(a_cov_data)
                all_cov_data_0.append(a_cov_data)
            else:
                break
    with open("output/1.txt", 'r') as fr:
        while True:
            oneline = fr.readline()
            if oneline:
                oneline = oneline.rstrip()
                a_cov_data = oneline.split(' ')
                for item in range(len(a_cov_data)):
                    a_cov_data[item] = float(a_cov_data[item])
                # print(a_cov_data)
                all_cov_data_1.append(a_cov_data)
            else:
                break
    all_cov_data.append(all_cov_data_0)
    all_cov_data.append(all_cov_data_1)
    return all_cov_data

#存入平均相对差异
def saveTxt(all_cov_data_0,all_cov_data_1):
    with open("output/mean_diff_0.txt",'w')as fw:
        for line in all_cov_data_0:
            fw.write(str(line)+'\n')
    with open("output/mean_diff_1.txt",'w')as fw:
        for line in all_cov_data_1:
            fw.write(str(line)+'\n')
#画图
def drawGraph():

    all_cov_data=txtFileRead()
    all_cov_data_0=[]
    all_cov_data_1=[]
    print(all_cov_data[0])
    print(all_cov_data[1])

    for item in all_cov_data[0]:
        all_cov_data_0.append(mean_diff(item))
    print(all_cov_data_0)
    for item in all_cov_data[1]:
        all_cov_data_1.append(mean_diff(item))
    print(all_cov_data_1)
    #保存数据
    saveTxt(all_cov_data_0,all_cov_data_1)

    #画图

    plt.figure('0 1')
    # plt.plot(all_cov_data_0)
    plt.ylabel("Mean relative difference")
    l0,=plt.plot(all_cov_data_0,color='red')
    l1,=plt.plot(all_cov_data_1)
    plt.ylim((0,0.3))
    plt.legend(handles=[l0,l1], labels=['0',1],loc='best')
    plt.show()


    plt.figure('pdf')
    plt.xlabel('Mean relative difference')
    plt.ylabel('frequency')
    hist,bin_edges=np.histogram(all_cov_data_0,bins=500)
    l0,=plt.plot(bin_edges[:500],hist,color='red')
    #plt.plot(bin_edges[:10], hist, "o",color='red')
    print('hist',hist)
    print('bin_edge',bin_edges)
    hist1,bin_edges1=np.histogram(all_cov_data_1,bins=500)
    l1,=plt.plot(bin_edges1[:500],hist1)
    plt.legend(handles=[l0, l1], labels=['0', 1], loc='best')
    # plt.xticks(bin_edges)
    # plt.hist(arr,bin_edges)
    print(hist)
    print(bin_edges)
    plt.show()

    plt.figure('cdf')
    plt.xlabel('Mean relative difference')

    hist,bin_edges=np.histogram(all_cov_data_0,bins=500)
    hist1, bin_edges1 = np.histogram(all_cov_data_1,bins=500)
    cdf_0=np.cumsum(hist)
    cdf_1=np.cumsum(hist1)
    l0,=plt.plot(bin_edges[:500],cdf_0/sum(hist),color='red')
    l1,=plt.plot(bin_edges1[:500],cdf_1/sum(hist1))
    plt.legend(handles=[l0, l1], labels=['0', 1], loc='best')
    plt.show()

if __name__ == "__main__":
    drawGraph()