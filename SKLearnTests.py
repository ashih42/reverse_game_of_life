import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ~--------------------------------------------------------------~
# NOTES

# Source: https://stats.stackexchange.com/questions/14761/learning-multiple-output
#         https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


# how to run andys stuff:
# run setup script
# python train.py train.csv
# half_stride to change surrounding cells
# change num features
# python predict.py param.dat test.csv
# export RGOL_SAVE_PLOTS, RGOL_VERBOSE, RGOL_SHOW_PLOTS
# ~--------------------------------------------------------------~

def holdout(data, ratio = 0.7):
    d = data.copy(deep = True)
    d = d.sample(frac = 1)
    train_count = math.floor(d.shape[0] * ratio)
    test_count = d.shape[0] - train_count
    train = d.head(train_count)
    test = d.tail(test_count)
    return train, test

def arr_diff(X, Y):
    R = np.subtract(X, Y)
    return R.sum()

def write_submission_file(X, filename='submission.csv'):
    with open(filename, 'w+') as submission_file:
        submission_file.write('id,start.1,start.2,start.3,start.4,start.5,start.6,start.7,start.8,start.9,start.10,start.11,start.12,start.13,start.14,start.15,start.16,start.17,start.18,start.19,start.20,start.21,start.22,start.23,start.24,start.25,start.26,start.27,start.28,start.29,start.30,start.31,start.32,start.33,start.34,start.35,start.36,start.37,start.38,start.39,start.40,start.41,start.42,start.43,start.44,start.45,start.46,start.47,start.48,start.49,start.50,start.51,start.52,start.53,start.54,start.55,start.56,start.57,start.58,start.59,start.60,start.61,start.62,start.63,start.64,start.65,start.66,start.67,start.68,start.69,start.70,start.71,start.72,start.73,start.74,start.75,start.76,start.77,start.78,start.79,start.80,start.81,start.82,start.83,start.84,start.85,start.86,start.87,start.88,start.89,start.90,start.91,start.92,start.93,start.94,start.95,start.96,start.97,start.98,start.99,start.100,start.101,start.102,start.103,start.104,start.105,start.106,start.107,start.108,start.109,start.110,start.111,start.112,start.113,start.114,start.115,start.116,start.117,start.118,start.119,start.120,start.121,start.122,start.123,start.124,start.125,start.126,start.127,start.128,start.129,start.130,start.131,start.132,start.133,start.134,start.135,start.136,start.137,start.138,start.139,start.140,start.141,start.142,start.143,start.144,start.145,start.146,start.147,start.148,start.149,start.150,start.151,start.152,start.153,start.154,start.155,start.156,start.157,start.158,start.159,start.160,start.161,start.162,start.163,start.164,start.165,start.166,start.167,start.168,start.169,start.170,start.171,start.172,start.173,start.174,start.175,start.176,start.177,start.178,start.179,start.180,start.181,start.182,start.183,start.184,start.185,start.186,start.187,start.188,start.189,start.190,start.191,start.192,start.193,start.194,start.195,start.196,start.197,start.198,start.199,start.200,start.201,start.202,start.203,start.204,start.205,start.206,start.207,start.208,start.209,start.210,start.211,start.212,start.213,start.214,start.215,start.216,start.217,start.218,start.219,start.220,start.221,start.222,start.223,start.224,start.225,start.226,start.227,start.228,start.229,start.230,start.231,start.232,start.233,start.234,start.235,start.236,start.237,start.238,start.239,start.240,start.241,start.242,start.243,start.244,start.245,start.246,start.247,start.248,start.249,start.250,start.251,start.252,start.253,start.254,start.255,start.256,start.257,start.258,start.259,start.260,start.261,start.262,start.263,start.264,start.265,start.266,start.267,start.268,start.269,start.270,start.271,start.272,start.273,start.274,start.275,start.276,start.277,start.278,start.279,start.280,start.281,start.282,start.283,start.284,start.285,start.286,start.287,start.288,start.289,start.290,start.291,start.292,start.293,start.294,start.295,start.296,start.297,start.298,start.299,start.300,start.301,start.302,start.303,start.304,start.305,start.306,start.307,start.308,start.309,start.310,start.311,start.312,start.313,start.314,start.315,start.316,start.317,start.318,start.319,start.320,start.321,start.322,start.323,start.324,start.325,start.326,start.327,start.328,start.329,start.330,start.331,start.332,start.333,start.334,start.335,start.336,start.337,start.338,start.339,start.340,start.341,start.342,start.343,start.344,start.345,start.346,start.347,start.348,start.349,start.350,start.351,start.352,start.353,start.354,start.355,start.356,start.357,start.358,start.359,start.360,start.361,start.362,start.363,start.364,start.365,start.366,start.367,start.368,start.369,start.370,start.371,start.372,start.373,start.374,start.375,start.376,start.377,start.378,start.379,start.380,start.381,start.382,start.383,start.384,start.385,start.386,start.387,start.388,start.389,start.390,start.391,start.392,start.393,start.394,start.395,start.396,start.397,start.398,start.399,start.400\n')
        for n in range(len(X)):
            submission_file.write(str(n + 1))
            for i in range(len(X[n])):
                submission_file.write(',')
                if X[n][i] > 0.5:
                    submission_file.write('1')
                else: 
                    submission_file.write('0')
            submission_file.write('\n')

# function that takes a board and returns 400 sets of cells surrounding

def neighboring_cells(board, x, y, surr=7):
    result = []
    rng_l = -surr // 2 + 1
    rng_h = rng_l + surr
    for i in range(rng_l, rng_h):
        for j in range(rng_l, rng_h):
            if x + i < board.shape[0] and x + i >= 0 and y + j < board.shape[0] and y + j >= 0:
                result.append(board[x + i][y + j])
            else:
                result.append(0)
    return result
    #for i in range(-2, 3):
#

def fooify(X, y, surr=7):
    result_x = []
    result_y = []
    cells = []
    X_y = X.tolist()
    epochs = X_y.pop(0)
    X = np.array(X_y)
    X = np.reshape(X, (20, 20))
    for i in range(len(X)):
        for j in range(len(X[i])):
            cells = neighboring_cells(X, i, j, surr)
            cells.insert(0, epochs)
            result_x.append(cells)
            result_y.append(X[i][j])
    return result_x, result_y

def fooify2(X, y, surr=7):
    result_x = []
    result_y = []
    cells = []
    X = np.reshape(X, (20, 20))
    y = np.reshape(y, (20, 20))
    for i in range(len(X)):
        for j in range(len(X[i])):
            cells = neighboring_cells(X, i, j, surr)
            result_x.append(cells)
            result_y.append(y[i][j])
    return result_x, result_y

def get_diff(X, Y):
    result = 0
    assert(len(X) == len(Y))
    for i in range(len(X)):
        if round(X[i]) != round(Y[i]):
            result += 1
    return result

            
# X, y

# board_in, board_out
# cell+surrounding, cells

# Test 7: One model for each cell and epoch. In total: 2000 models
# Slow. Does worse than one model per epoch looking at surrounding cells

def test_7():
    print('Reading training data...')
    data = pd.read_csv("repo/resources/train.csv")
    training, testing = holdout(data)

    model_list = []

    surr_cells = 7
    training_count = 1000
    testing_count = 1000

    foor = [1]
    for i in range(402, 802):
        foor.append(i)

    for i in range(5):
        curr_training = training.loc[training['delta'] == i + 1]
        train = curr_training.iloc[:training_count, range(402, 802)]
        expected = curr_training.iloc[:training_count, range(2, 402)]

        X = train.as_matrix()
        y = expected.as_matrix()

        list_x, list_y = [], []
        cnt = 0
        for t, r in zip(X, y):
            print('Creating training set', i + 1, '...', cnt, '/', training_count)
            cnt += 1
            rx, ry = fooify2(t, r, surr_cells)
            list_x.extend(rx)
            list_y.extend(ry)

        mor_board = []
        for cell in range(400):
            print('Fitting ML algorithm to model', i + 1, '...', cell, '/ 400')
            mor = RandomForestClassifier(n_estimators=100, random_state=0)
            tx = list_x[cell::400]
            ty = list_y[cell::400]
            mor.fit(tx, ty)
            mor_board.append(mor)

        model_list.append(mor_board)


    valid_in = testing.iloc[:testing_count, foor].values
    valid_out = testing.iloc[:testing_count, range(2, 402)].values

    result = 0
    cnt = 0
    for dt, ex in zip(valid_in, valid_out):
        X_y = dt.tolist()
        epochs = X_y.pop(0)
        ddt = np.array(X_y)
        ddt = np.reshape(ddt, (20, 20))
        test_y = []
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(ddt, i, j, surr_cells)
                test_y.append(vrr)
        dummy = []
        for i, entry in enumerate(test_y):
            mor = model_list[epochs - 1][i]
            V = mor.predict([entry])
            dummy.extend(V.flatten())
        count = get_diff(ex, dummy)
        result += count / 400
        print('Scoring...', cnt + 1, '=', count / 400, 'avg =', result / (cnt + 1))
        cnt += 1


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# Test 6: One model per epoch. Model predicts if a cell will be alive based on surronding tiles
# Best one yet. About a 87.5% accuracy

def test_6():
    print('Reading training data...')
    training = pd.read_csv("repo/resources/train.csv")
    print('Reading testing data...')
    testing = pd.read_csv("repo/resources/test.csv")

    model_list = []

    surr_cells = 7
    training_count = 1000
    testing_count = testing.shape[0]

    vx = testing.iloc[:testing_count, range(1, 402)].values

    for i in range(5):
        mor = RandomForestClassifier(n_estimators=100, random_state=0)

        curr_training = training.loc[training['delta'] == i + 1]
        train = curr_training.iloc[:training_count, range(402, 802)]
        expected = curr_training.iloc[:training_count, range(2, 402)]

        X = train.as_matrix()
        y = expected.as_matrix()

        list_x, list_y = [], []
        cnt = 0
        for t, r in zip(X, y):
            print('Creating training set', i + 1, '...', cnt, '/', training_count)
            cnt += 1
            rx, ry = fooify2(t, r, surr_cells)
            list_x.extend(rx)
            list_y.extend(ry)
        
        print('Fitting ML algorithm to model', i + 1, '...')
        mor.fit(list_x, list_y)
        model_list.append(mor)

    cnt = 0
    finish = []
    dummy = []
    for dt in vx:
        print('Writing predictions...', cnt, '/', testing_count)
        X_y = dt.tolist()
        epochs = X_y.pop(0)
        ddt = np.array(X_y)
        ddt = np.reshape(ddt, (20, 20))
        test_y = []
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(ddt, i, j, surr_cells)
                test_y.append(vrr)
        mor = model_list[epochs - 1]
        V = mor.predict(test_y)
        dummy = V.flatten()
        finish.append(dummy)
        cnt += 1
    write_submission_file(finish)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# Same as test 6, only not writing to file

def test_5():
    print('Reading training data...')
    data = pd.read_csv("repo/resources/train.csv")
    #training = data.head(100)
    #testing = data.tail(1)
    training, testing = holdout(data)
    #training = data.iloc[:20, range(0, 802)]
    #testing = data.iloc[25:26, range(0, 802)]

    model_list = []

    surr_cells = 7
    training_count = 1000
    testing_count = 1000

    foor = [1]
    for i in range(402, 802):
        foor.append(i)

    for i in range(5):
        mor = RandomForestClassifier(n_estimators=100, random_state=0) # 0.129
        #mor = RandomForestRegressor(n_estimators=100, random_state=0)
        #mor = GradientBoostingClassifier(n_estimators=100, random_state=0) # 0.134
        #mor = LogisticRegression()

        curr_training = training.loc[training['delta'] == i + 1]
        train = curr_training.iloc[:training_count, range(402, 802)]
        expected = curr_training.iloc[:training_count, range(2, 402)]

        X = train.as_matrix()
        y = expected.as_matrix()

        list_x, list_y = [], []
        cnt = 0
        for t, r in zip(X, y):
            print('Creating training set', i + 1, '...', cnt, '/', training_count)
            cnt += 1
            rx, ry = fooify2(t, r, surr_cells)
            list_x.extend(rx)
            list_y.extend(ry)
        
        print('Fitting ML algorithm to model', i + 1, '...')
        mor.fit(list_x, list_y)
        model_list.append(mor)


    valid_in = testing.iloc[:testing_count, foor].values
    valid_out = testing.iloc[:testing_count, range(2, 402)].values

    result = 0
    cnt = 0
    for dt, ex in zip(valid_in, valid_out):
        X_y = dt.tolist()
        epochs = X_y.pop(0)
        ddt = np.array(X_y)
        ddt = np.reshape(ddt, (20, 20))
        test_y = []
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(ddt, i, j, surr_cells)
                test_y.append(vrr)
        mor = model_list[epochs - 1]
        V = mor.predict(test_y)
        dummy = V.flatten()
        #print(ex)
        #print(dummy)
        count = get_diff(ex, dummy)
        result += count / 400
        print('Scoring...', cnt + 1, '=', count / 400, 'avg =', result / (cnt + 1))
        cnt += 1
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def test_4():
    print('Reading training data...')
    data = pd.read_csv("repo/resources/train.csv")
    
    datas = [1]
    for i in range(402, 802):
        datas.append(i)

    training, testing = holdout(data)

    surr_cells = 7
    training_count = 100
    testing_count = 100
    mor = RandomForestClassifier(n_estimators=100, random_state=0)

    train = training.iloc[:training_count, datas]
    expected = training.iloc[:training_count, range(2, 402)]

    valid_in = testing.iloc[:testing_count, datas].values
    valid_out = testing.iloc[:testing_count, range(2, 402)].values

    X = train.as_matrix()
    y = expected.as_matrix()

    list_x, list_y = [], []
    cnt = 0
    for t, r in zip(X, y):
        print('Creating training set... ', cnt, '/', training_count)
        t.pop(0)
        cnt += 1
        rx, ry = fooify2(t, r, surr_cells)
        list_x.extend(rx)
        list_y.extend(ry)

    print('Fitting ML algorithm...')
    mor.fit(list_x, list_y)

    test_y = []
    inc = 0

    for exp in valid_in:
        print('Getting expected test data... ', inc, '/', testing_count)
        inc += 1
        X_y = exp.tolist()
        epochs = X_y.pop(0)
        dex = np.array(X_y)
        dex = np.reshape(dex, (20, 20))
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(dex, i, j, surr_cells)
                vrr.insert(0, epochs)
                test_y.append(vrr)

    print('Predicting...')
    V = mor.predict(test_y)
    finish = []
    dummy = []
    result = 0
    for i in range(len(V)):
        if i % 400 == 0 and i > 0:
            #print(valid_out[i // 400 - 1], dummy)
            count = get_diff(valid_out[i // 400 - 1], dummy)
            finish.append(dummy)
            result += count / 400
            print('Scoring... ', i // 400, '=', count / 400, ', avg =', result / (i // 400))
            dummy = []
        dummy.append(V[i])
    finish.append(dummy)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def test_3():
    print('Reading training data...')
    data = pd.read_csv("repo/resources/train.csv")
    print('Reading testing data...')
    testing = pd.read_csv("repo/resources/test.csv")
    datas = [1]
    for i in range(402, 802):
        datas.append(i)

    training, testing = holdout(data, ratio=0.7)
    
    surr_cells = 7
    training_count = 5000
    testing_count = 1000
    mor = RandomForestClassifier(n_estimators=100, random_state=0)
    #mor = RandomForestRegressor(n_estimators=100, random_state=0)
    #mor = LogisticRegression(random_state=0)
    #mor = ExtraTreesClassifier(n_estimators=100, random_state=0)
    #mor = GradientBoostingRegressor(random_state=0)

    train = training.iloc[:training_count, datas]
    expected = training.iloc[:training_count, range(2, 402)]

    valid_in = testing.iloc[:testing_count, datas].values
    valid_out = testing.iloc[:testing_count, range(2, 402)].values

    X = train.as_matrix()
    y = expected.as_matrix()

    list_x, list_y = [], []
    cnt = 0
    for t, r in zip(X, y):
        print('Creating training set... ', cnt, '/', training_count)
        cnt += 1
        rx, ry = fooify(t, r, surr_cells)
        list_x.extend(rx)
        list_y.extend(ry)
    
    print('Fitting ML algorithm...')
    mor.fit(list_x, list_y)

    test_y = []
    inc = 0

    for exp in valid_in:
        print('Getting expected test data... ', inc, '/', testing_count)
        inc += 1
        X_y = exp.tolist()
        epochs = X_y.pop(0)
        dex = np.array(X_y)
        dex = np.reshape(dex, (20, 20))
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(dex, i, j, surr_cells)
                vrr.insert(0, epochs)
                test_y.append(vrr)

    print('Predicting...')
    V = mor.predict(test_y)
    finish = []
    dummy = []
    result = 0
    for i in range(len(V)):
        if i % 400 == 0 and i > 0:
            #print(valid_out[i // 400 - 1], dummy)
            count = get_diff(valid_out[i // 400 - 1], dummy)
            finish.append(dummy)
            result += count / 400
            print('Scoring... ', i // 400, '=', count / 400, ', avg =', result / (i // 400))
            dummy = []
        dummy.append(V[i])
    finish.append(dummy)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def test_2():
    print('Reading training data...')
    data = pd.read_csv("repo/resources/train.csv")
    print('Reading testing data...')
    testing = pd.read_csv("repo/resources/test.csv")
    datas = [1]
    for i in range(402, 802):
        datas.append(i)
    
    surr_cells = 5
    training_count = 1000
    testing_count = testing.shape[0]
    #mor = RandomForestClassifier(n_estimators=100)
    mor = GradientBoostingRegressor(random_state=0)


    training = data
    train = training.iloc[:training_count, datas]
    expected = training.iloc[:training_count, range(2, 402)]

    X = train.as_matrix()
    y = expected.as_matrix()

    list_x, list_y = [], []

    cnt = 0
    for t, r in zip(X, y):
        print('Creating training set... ', cnt, '/', training_count)
        cnt += 1
        rx, ry = fooify(t, r, surr_cells)
        list_x.extend(rx)
        list_y.extend(ry)

    print('Fitting ML algorithm...')
    mor.fit(list_x, list_y)

    vx = testing.iloc[:testing_count, range(1, 402)].values
    test_y = []
    inc = 0

    for exp in vx:
        print('Getting expected test data... ', inc, '/', testing_count)
        inc += 1
        X_y = exp.tolist()
        epochs = X_y.pop(0)
        dex = np.array(X_y)
        dex = np.reshape(dex, (20, 20))
        for i in range(20):
            for j in range(20):
                vrr = neighboring_cells(dex, i, j, surr_cells)
                vrr.insert(0, epochs)
                test_y.append(vrr)

    print('Predicting...')
    V = mor.predict(test_y)
    finish = []
    dummy = []

    for i in range(len(V)):
        if i % 400 == 0 and i > 0:
            print('Writing predictions... ', i // 400, '/', testing_count)
            finish.append(dummy)
            dummy = []
        dummy.append(V[i])
    finish.append(dummy)
    write_submission_file(finish, filename='submission')

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\      

def test_and_submission():
    data = pd.read_csv("repo/resources/train.csv")
    testing = pd.read_csv("repo/resources/test.csv")
    datas = [1]
    for i in range(402, 802):
        datas.append(i)

    training = data

    train = training.iloc[:training.shape[0], datas]
    expected = training.iloc[:training.shape[0], range(2, 402)]

    valid_in = testing.iloc[:testing.shape[0], range(1, 402)].values

    mor = RandomForestClassifier(n_estimators=100)

    X = train.as_matrix()
    y = expected.as_matrix()
    X_val = valid_in.as_matrix()

    X = X[0]
    y = y[0]
    X_val = X_val[0]

    mor.fit(X, y)
    X = mor.predict(X_val)
    write_submission_file(X)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def test_1():
    data = pd.read_csv("repo/resources/train100.csv")
    testing = pd.read_csv("repo/resources/test100.csv")
    datas = [1]
    for i in range(402, 802):
        datas.append(i)

    training, testing = holdout(data, ratio=0.9)

    train = training.iloc[:training.shape[0], datas]
    expected = training.iloc[:training.shape[0], range(2, 402)]

    valid_in = testing.iloc[:testing.shape[0], datas].values
    valid_out = testing.iloc[:testing.shape[0], range(2, 402)].values

    mor = RandomForestClassifier(n_estimators=100)

    X = np.array(train)
    y = np.array(expected)

    X_val = np.array(valid_in)
    y_val = np.array(valid_out)

    mor.fit(X, y)
    X = mor.predict(X_val)
    result = 0
    for i in range(len(X)):
        c = X[i].copy()
        for x in range(len(c)):
            c[x] = c[x] > 0.5
        R = np.subtract(y_val[i], c)
        count = R.sum()
        result += count / 400
        if (i % 1 == 0):
            grog = i
            if grog == 0:
                grog = 1
            print(i, ' = ', count / 400, ', avg = ', result / grog)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def main():
    test_5()

if __name__ == '__main__':
    main()