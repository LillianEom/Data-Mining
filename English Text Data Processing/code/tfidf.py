# -*- coding:utf-8 -*-
import os
import string
import nltk
import datetime
import math

# function: 判断字符是否为英文字母
def isMessy(str):
    for c in str:
        if c < 'A':
            return True
        if c > 'Z' and c < 'a':
            return True
        if c > 'z':
            return True
    return False

# function: 数据预处理（分词、数据清洗），生成文章的词频表
def preprocess(inputUrl,outputUrl):
    # 二进制方式打开文件，否则读到奇怪字符时会停止
    f = open(inputUrl,'rb')
    raw = f.read()
    f.close()
    # 删除标点符号和数字
    pun_num = string.punctuation + string.digits
    identity = string.maketrans(' ', ' ')
    raw = raw.translate(identity,pun_num)
    # 转为小写字母
    raw = raw.lower()
    # 分词
    list = nltk.word_tokenize(raw)

    # 删除长度小于3的单词，删除英文字母以外的字符，删除stop words，删除非英文单词
    from nltk.corpus import stopwords
    stopWordsList = stopwords.words('english')
    fopen = open('stopwords.txt')
    for line in fopen:
        stopWordsList.append(line[:-1])
    stopWordsSet = set(stopWordsList)
    englishWordList = nltk.corpus.words.words()
    englishWordSet = set(englishWordList)
    i = 0
    while i < len(list):
        if len(list[i]) < 3 or isMessy(list[i]) or list[i] in stopWordsSet or list[i] not in englishWordSet:
            del list[i]
            continue
        i = i + 1
    # 取词干
    wnl = nltk.stem.WordNetLemmatizer()
    for i in range(len(list)):
        list[i] = wnl.lemmatize(list[i])
    # 再次删除长度小于3的单词
    i = 0
    while i < len(list):
        if len(list[i]) < 3 :
            del list[i]
            continue
        i = i + 1
    # 统计词频
    wordDict = {}
    for i in range(len(list)):
        if (wordDict.has_key(list[i])):
            wordDict[list[i]] = wordDict[list[i]] + 1
        else:
            wordDict[list[i]] = 1
    # 输出词频表到文件
    output_file = open(outputUrl, 'w')
    for key in wordDict:
        output_file.write(key+' '+str(wordDict[key]) + '\n')
    output_file.close()

# function: 遍历语料库，生成所有文章的词频表，返回文章总数
def calWordListForPaper(rootdir):
    fileSum = 0
    class_str = ''
    i = 0
    for parent, dirnames, filenames in os.walk(rootdir):    # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        j = 0
        for filename in filenames:
            inputUrl = os.path.join(parent, filename)
            if os.path.splitext(filename)[1] != '.txt':
                continue
            fileSum=fileSum+1
            if(i<10):
                class_str = '0'+str(i)
            else:
                class_str = str(i)
            outputUrl='data\\'+class_str + filename
            preprocess(inputUrl,outputUrl)
            j = j + 1
        i = i + 1
    return fileSum

# function: 从文件读取一篇文章的词频表，保存在一个字典中并返回
def readWordCntDict(fileurl):
    f = open(fileurl)
    worddict = {}
    for line in f:
        item = line.split()
        worddict[item[0]] = string.atoi(item[1])
    f.close()
    return worddict


# 主程序
starttime = datetime.datetime.now()
sourceRoot = "E:\DataMining\Assignment1\ICML"
class7Url = sourceRoot+r"\07. Kernel Methods"
class7FilenameOutput = 'Class7FileName.txt'
wordListRoot = "data"
WOutput = 'W.txt'
resultOutput = 'result.txt'

P = calWordListForPaper(sourceRoot)
# 读取所有文件名
filenameList = []
fileUrlList = []
for parent, dirnames, filenames in os.walk(wordListRoot):
    for filename in filenames:
        if os.path.splitext(filename)[1] != '.txt':
            continue
        filenameList.append(filename)
        fileUrlList.append(os.path.join(parent, filename))

# 提取Class7的词频表文件名，输出到一个文件中
output_file = open(class7FilenameOutput, 'w')
for parent, dirnames, filenames in os.walk(class7Url):
    for filename in filenames:
        if os.path.splitext(filename)[1] != '.txt':
            continue
        output_file.write('07'+filename+'\n')
output_file.close()

# 生成字典{文件名：词频表} 词频表也是字典{单词：词频}
fileDict = {}
for item in fileUrlList:
    fileDict[item[5:]]=readWordCntDict(item)    # 去掉data\\，字典中的键只存文件名

# 统计所有文件中的单词，按字典序排序后存入List，另输入到一个文件中作为结果的一部分
wordsList = []
for item in filenameList:
    dict = fileDict[item]
    wordsList=wordsList+dict.keys()
allWordSet = set(wordsList)        #去重
allWordList = list(allWordSet)
allWordList.sort()                  #排序
allWordSet = set(allWordList)
W = len(allWordList)
output_file = open(WOutput, 'w')
for word in allWordList:
    output_file.write(word+'\n')
output_file.close()

# 计算所有文件所有单词的TF值，TF是字典{文件名：TF值的List}，List的顺序与W一一对应
TF = {}
for file in fileDict:
    rowTF = []
    wordDict = fileDict[file]
    wordSum = sum(wordDict.values())
    for word in allWordList:
        if word in wordDict:
            wordFreq = wordDict[word]
        else:
            wordFreq = 0
        tf = float(wordFreq)/float(wordSum)
        rowTF.append(tf)
    TF[file]=rowTF

# 计算每个单词在多少文件中出现，字典{单词：包含它的文件数}
paperCnt = {}
for word in allWordList:
    paperNum = 0
    for file in fileDict:
        wordDict = fileDict[file]
        if word in wordDict:
            paperNum=paperNum+1
    paperCnt[word] = paperNum

# 计算每个单词的IDF，字典{单词：idf}
IDF = {}
for word in allWordList:
    idf = math.log10(float(P)/float(paperCnt[word]))
    IDF[word] = idf

# 计算最终的TF-IDF矩阵，TF-IDF是一个字典{文件名：特征向量}
TFIDF = {}
for file in fileDict:
    row = []
    for j in range(0, W):
        word = allWordList[j]
        row.append(TF[file][j]*IDF[word])
    TFIDF[file]=row

# 从文件中读取Class7的词频表文件名
class7fileList = []
f = open(class7FilenameOutput)
for line in f:
    class7fileList.append(line)
f.close()

# 抽取Class7的特征矩阵
result7 = []
for file in class7fileList:
    file=file[:-1]               #去掉末尾\n
    result7.append(TFIDF[file])

# 以稀疏矩阵的表达方式，格式化输出Class7的特征矩阵
output_file = open(resultOutput, 'w')
for i in range(0,26):
    row = result7[i]
    output_file.write(str(i+1)+'\n'+'[')
    for j in range(0, W):
        if row[j] != 0:
            output_file.write("%5d:%10.8f," % (j+1,row[j]) )
    output_file.write(']'+ '\n')
output_file.close()
endtime = datetime.datetime.now()

# 输出程序运行时间
print 'running time: '+ str((endtime - starttime).seconds) + 's'