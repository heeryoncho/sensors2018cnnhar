import numpy as np
import pandas as pd
import os


def generate_x(x_file):
    with open(x_file) as f:
        container = f.readlines()
    result = []
    for line in container:
        tmp1 = line.strip()
        tmp2 = tmp1.replace('  ', ' ')
        # print tmp2
        tmp_ary = map(float, tmp2.split(' '))
        # nan_count = sum(math.isnan(x) for x in tmp_ary)
        # if (tmp_ary[243] != 0.0) & (nan_count < 10):
        if (tmp_ary[243] != 0.0):
            result.append(tmp_ary)
    return np.array(result)

def sel_columns_raw(df):
    # Raw denotes lower body sensor data.
    # See 'def sel_columns_lower(df)'.
    knee_1 = df[[1, 2, 3]].values.flatten()
    knee_2 = df[[19, 20, 21]].values.flatten()
    hip = df[[4, 5, 6]].values.flatten()

    left_1 = df[[102, 103, 104]].values.flatten()
    left_2 = df[[105, 106, 107]].values.flatten()
    left_3 = df[[108, 109, 110]].values.flatten()
    left_4 = df[[111, 112, 113]].values.flatten()
    left_5 = df[[114, 115, 116]].values.flatten()
    right_1 = df[[118, 119, 120]].values.flatten()
    right_2 = df[[121, 122, 123]].values.flatten()
    right_3 = df[[124, 125, 126]].values.flatten()
    right_4 = df[[127, 128, 129]].values.flatten()
    right_5 = df[[130, 131, 132]].values.flatten()

    data = np.concatenate((knee_1, knee_2, hip,
                           left_1, left_2, left_3, left_4, left_5,
                           right_1, right_2, right_3, right_4, right_5))
    # print data.shape
    return data

def sel_columns_upper(df):
    lua_up = df[[7, 8, 9]]
    lua_bottom = df[[28, 29, 30]]

    lua_1 = df[[76, 77, 78]]
    lua_2 = df[[79, 80, 81]]
    lua_3 = df[[82, 83, 84]]

    lla_1 = df[[89, 90, 91]]
    lla_2 = df[[92, 93, 94]]
    lla_3 = df[[95, 96, 97]]

    lwr = df[[31, 32, 33]]
    lh = df[[13, 14, 15]]

    rua_up = df[[25, 26, 27]]
    rua_bottom = df[[10, 11, 12]]

    rua_1 = df[[50, 51, 52]]
    rua_2 = df[[53, 54, 55]]
    rua_3 = df[[56, 57, 58]]

    rla_1 = df[[63, 64, 65]]
    rla_2 = df[[66, 67, 68]]
    rla_3 = df[[69, 70, 71]]

    #rwr = df[[22, 23, 24]]
    #rh = df[[34, 35, 36]]

    lua_up_mean = np.mean(lua_up, axis=0)
    lua_up_std = np.std(lua_up, axis=0)
    lua_up_min = np.min(lua_up, axis=0)
    lua_up_max = np.max(lua_up, axis=0)
    lua_bottom_mean = np.mean(lua_bottom, axis=0)
    lua_bottom_std = np.std(lua_bottom, axis=0)
    lua_bottom_min = np.min(lua_bottom, axis=0)
    lua_bottom_max = np.max(lua_bottom, axis=0)
    lua_1_mean = np.mean(lua_1, axis=0)
    lua_1_std = np.std(lua_1, axis=0)
    lua_1_min = np.min(lua_1, axis=0)
    lua_1_max = np.max(lua_1, axis=0)
    lua_2_mean = np.mean(lua_2, axis=0)
    lua_2_std = np.std(lua_2, axis=0)
    lua_2_min = np.min(lua_2, axis=0)
    lua_2_max = np.max(lua_2, axis=0)
    lua_3_mean = np.mean(lua_3, axis=0)
    lua_3_std = np.std(lua_3, axis=0)
    lua_3_min = np.min(lua_3, axis=0)
    lua_3_max = np.max(lua_3, axis=0)
    lla_1_mean = np.mean(lla_1, axis=0)
    lla_1_std = np.std(lla_1, axis=0)
    lla_1_min = np.min(lla_1, axis=0)
    lla_1_max = np.max(lla_1, axis=0)
    lla_2_mean = np.mean(lla_2, axis=0)
    lla_2_std = np.std(lla_2, axis=0)
    lla_2_min = np.min(lla_2, axis=0)
    lla_2_max = np.max(lla_2, axis=0)
    lla_3_mean = np.mean(lla_3, axis=0)
    lla_3_std = np.std(lla_3, axis=0)
    lla_3_min = np.min(lla_3, axis=0)
    lla_3_max = np.max(lla_3, axis=0)
    lwr_mean = np.mean(lwr, axis=0)
    lwr_std = np.std(lwr, axis=0)
    lwr_min = np.min(lwr, axis=0)
    lwr_max = np.max(lwr, axis=0)
    lh_mean = np.mean(lh, axis=0)
    lh_std = np.std(lh, axis=0)
    lh_min = np.min(lh, axis=0)
    lh_max = np.max(lh, axis=0)
    rua_up_mean = np.mean(rua_up, axis=0)
    rua_up_std = np.std(rua_up, axis=0)
    rua_up_min = np.min(rua_up, axis=0)
    rua_up_max = np.max(rua_up, axis=0)
    rua_bottom_mean = np.mean(rua_bottom, axis=0)
    rua_bottom_std = np.std(rua_bottom, axis=0)
    rua_bottom_min = np.min(rua_bottom, axis=0)
    rua_bottom_max = np.max(rua_bottom, axis=0)
    rua_1_mean = np.mean(rua_1, axis=0)
    rua_1_std = np.std(rua_1, axis=0)
    rua_1_min = np.min(rua_1, axis=0)
    rua_1_max = np.max(rua_1, axis=0)
    rua_2_mean = np.mean(rua_2, axis=0)
    rua_2_std = np.std(rua_2, axis=0)
    rua_2_min = np.min(rua_2, axis=0)
    rua_2_max = np.max(rua_2, axis=0)
    rua_3_mean = np.mean(rua_3, axis=0)
    rua_3_std = np.std(rua_3, axis=0)
    rua_3_min = np.min(rua_3, axis=0)
    rua_3_max = np.max(rua_3, axis=0)
    rla_1_mean = np.mean(rla_1, axis=0)
    rla_1_std = np.std(rla_1, axis=0)
    rla_1_min = np.min(rla_1, axis=0)
    rla_1_max = np.max(rla_1, axis=0)
    rla_2_mean = np.mean(rla_2, axis=0)
    rla_2_std = np.std(rla_2, axis=0)
    rla_2_min = np.min(rla_2, axis=0)
    rla_2_max = np.max(rla_2, axis=0)
    rla_3_mean = np.mean(rla_3, axis=0)
    rla_3_std = np.std(rla_3, axis=0)
    rla_3_min = np.min(rla_3, axis=0)
    rla_3_max = np.max(rla_3, axis=0)
    #rwr_mean = np.mean(rwr, axis=0)
    #rwr_std = np.std(rwr, axis=0)
    #rwr_min = np.min(rwr, axis=0)
    #rwr_max = np.max(rwr, axis=0)
    #rh_mean = np.mean(rh, axis=0)
    #rh_std = np.std(rh, axis=0)
    #rh_min = np.min(rh, axis=0)
    #rh_max = np.max(rh, axis=0)

    data = np.concatenate((lua_up_mean, lua_up_std, lua_up_min, lua_up_max,
                           lua_bottom_mean, lua_bottom_std, lua_bottom_min, lua_bottom_max,
                           lua_1_mean, lua_1_std, lua_1_min, lua_1_max,
                           lua_2_mean, lua_2_std, lua_2_min, lua_2_max,
                           lua_3_mean, lua_3_std, lua_3_min, lua_3_max,
                           lla_1_mean, lla_1_std, lla_1_min, lla_1_max,
                           lla_2_mean, lla_2_std, lla_2_min, lla_2_max,
                           lla_3_mean, lla_3_std, lla_3_min, lla_3_max,
                           lwr_mean, lwr_std, lwr_min, lwr_max,
                           lh_mean, lh_std, lh_min, lh_max,
                           rua_up_mean, rua_up_std, rua_up_min, rua_up_max,
                           rua_bottom_mean, rua_bottom_std, rua_bottom_min, rua_bottom_max,
                           rua_1_mean, rua_1_std, rua_1_min, rua_1_max,
                           rua_2_mean, rua_2_std, rua_2_min, rua_2_max,
                           rua_3_mean, rua_3_std, rua_3_min, rua_3_max,
                           rla_1_mean, rla_1_std, rla_1_min, rla_1_max,
                           rla_2_mean, rla_2_std, rla_2_min, rla_2_max,
                           rla_3_mean, rla_3_std, rla_3_min, rla_3_max,
                           #rwr_mean, rwr_std, rwr_min, rwr_max,
                           #rh_mean, rh_std, rh_min, rh_max
                           ))
    '''

    The following three features were removed due to many missing values.

    rwr : right wrist
    rh : right hand

    Executed the following command to find missing value features/sensors:

    > nan_valid = np.argwhere(np.isnan(X_valid))
    > np.unique(nan[:, 1])

    '''
    # print data.shape
    return data

def sel_columns_lower(df):
    knee_1 = df[[1, 2, 3]]
    knee_2 = df[[19, 20, 21]]
    hip = df[[4, 5, 6]]

    left_1 = df[[102, 103, 104]]
    left_2 = df[[105, 106, 107]]
    left_3 = df[[108, 109, 110]]
    left_4 = df[[111, 112, 113]]
    left_5 = df[[114, 115, 116]]
    right_1 = df[[118, 119, 120]]
    right_2 = df[[121, 122, 123]]
    right_3 = df[[124, 125, 126]]
    right_4 = df[[127, 128, 129]]
    right_5 = df[[130, 131, 132]]

    knee_1_mean = np.mean(knee_1, axis=0)
    knee_1_std = np.std(knee_1, axis=0)
    knee_1_min = np.min(knee_1, axis=0)
    knee_1_max = np.max(knee_1, axis=0)
    knee_2_mean = np.mean(knee_2, axis=0)
    knee_2_std = np.std(knee_2, axis=0)
    knee_2_min = np.min(knee_2, axis=0)
    knee_2_max = np.max(knee_2, axis=0)
    hip_mean = np.mean(hip, axis=0)
    hip_std = np.std(hip, axis=0)
    hip_min = np.min(hip, axis=0)
    hip_max = np.max(hip, axis=0)
    left_1_mean = np.mean(left_1, axis=0)
    left_1_std = np.std(left_1, axis=0)
    left_1_min = np.min(left_1, axis=0)
    left_1_max = np.max(left_1, axis=0)
    left_2_mean = np.mean(left_2, axis=0)
    left_2_std = np.std(left_2, axis=0)
    left_2_min = np.min(left_2, axis=0)
    left_2_max = np.max(left_2, axis=0)
    left_3_mean = np.mean(left_3, axis=0)
    left_3_std = np.std(left_3, axis=0)
    left_3_min = np.min(left_3, axis=0)
    left_3_max = np.max(left_3, axis=0)
    left_4_mean = np.mean(left_4, axis=0)
    left_4_std = np.std(left_4, axis=0)
    left_4_min = np.min(left_4, axis=0)
    left_4_max = np.max(left_4, axis=0)
    left_5_mean = np.mean(left_5, axis=0)
    left_5_std = np.std(left_5, axis=0)
    left_5_min = np.min(left_5, axis=0)
    left_5_max = np.max(left_5, axis=0)
    right_1_mean = np.mean(right_1, axis=0)
    right_1_std = np.std(right_1, axis=0)
    right_1_min = np.min(right_1, axis=0)
    right_1_max = np.max(right_1, axis=0)
    right_2_mean = np.mean(right_2, axis=0)
    right_2_std = np.std(right_2, axis=0)
    right_2_min = np.min(right_2, axis=0)
    right_2_max = np.max(right_2, axis=0)
    right_3_mean = np.mean(right_3, axis=0)
    right_3_std = np.std(right_3, axis=0)
    right_3_min = np.min(right_3, axis=0)
    right_3_max = np.max(right_3, axis=0)
    right_4_mean = np.mean(right_4, axis=0)
    right_4_std = np.std(right_4, axis=0)
    right_4_min = np.min(right_4, axis=0)
    right_4_max = np.max(right_4, axis=0)
    right_5_mean = np.mean(right_5, axis=0)
    right_5_std = np.std(right_5, axis=0)
    right_5_min = np.min(right_5, axis=0)
    right_5_max = np.max(right_5, axis=0)
    data = np.concatenate((knee_1_mean, knee_1_std, knee_1_min, knee_1_max,
                           knee_2_mean, knee_2_std, knee_2_min, knee_2_max,
                           hip_mean, hip_std, hip_min, hip_max,
                           right_1_mean, right_1_std, right_1_min, right_1_max,
                           right_2_mean, right_2_std, right_2_min, right_2_max,
                           right_3_mean, right_3_std, right_3_min, right_3_max,
                           right_4_mean, right_4_std, right_4_min, right_4_max,
                           right_5_mean, right_5_std, right_5_min, right_5_max,
                           left_1_mean, left_1_std, left_1_min, left_1_max,
                           left_2_mean, left_2_std, left_2_min, left_2_max,
                           left_3_mean, left_3_std, left_3_min, left_3_max,
                           left_4_mean, left_4_std, left_4_min, left_4_max,
                           left_5_mean, left_5_std, left_5_min, left_5_max))
    # print data.shape
    return data

# Make new directory under '../data/OpportunityUCIDataset' folder

processed_dir = "../data/OpportunityUCIDataset/processed"
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)


# Generate training / validation / test data

# files_drill = ['S1-Drill.dat', 'S2-Drill.dat', 'S3-Drill.dat']   # We do not use Drill data.
files_train = ['S2-ADL1.dat', 'S2-ADL2.dat', 'S3-ADL1.dat', 'S3-ADL2.dat', 'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat']
files_valid = ['S2-ADL3.dat', 'S3-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat']
files_test = ['S2-ADL4.dat', 'S2-ADL5.dat', 'S3-ADL4.dat', 'S3-ADL5.dat']

data_type = ["raw", "upper", "lower"]
file_type = {"train": files_train, "valid": files_valid, "test": files_test}

dir_in = "/home/hcilab/Documents/HAR/Opportunity/OpportunityUCIDataset/dataset/"
dir_out = "../data/OpportunityUCIDataset/processed/"

for data_type_sel in data_type:
    print "------ {} data ------".format(data_type_sel)
    for k, v in file_type.iteritems():
        X = pd.DataFrame()
        y = pd.Series()
        for in_file in v:
            data = generate_x(dir_in + in_file)
            X = X.append(pd.DataFrame(data))

        print X.shape
        # print X.columns

        X_1 = X.loc[X[243] == 1.0]
        print "{}_101:".format(k), X_1.shape
        X_2 = X.loc[X[243] == 2.0]
        print "{}_102:".format(k), X_2.shape
        X_3 = X.loc[X[243] == 4.0]
        print "{}_104:".format(k), X_3.shape
        X_4 = X.loc[X[243] == 5.0]
        print "{}_105:".format(k), X_4.shape

        X_1 = X_1.fillna(method='ffill')
        # print X_1.isnull().sum().values
        X_2 = X_2.fillna(method='ffill')
        # print X_1.isnull().sum().values
        X_3 = X_3.fillna(method='ffill')
        # print X_1.isnull().sum().values
        X_4 = X_4.fillna(method='ffill')
        # print X_1.isnull().sum().values

        four_activities_data = [X_1, X_2, X_3, X_4]

        window = 15
        sliding = 7

        data_container = []
        label_container = []

        for activity in four_activities_data:
            for i in range(activity.shape[0] / sliding):
                df = activity.iloc[range((0 + i), (window + i))]
                label = df[243].astype(int).values
                label_container.append(label[-1])
                if data_type_sel is "raw":
                    data_container.append(sel_columns_raw(df))
                if data_type_sel is "upper":
                    data_container.append(sel_columns_upper(df))
                if data_type_sel is "lower":
                    data_container.append(sel_columns_lower(df))
                i = i + sliding

        print "{}_data:".format(k), np.asarray(data_container).shape
        print "{}_label:".format(k), np.asarray(label_container).shape

        np.save(dir_out + "{}_{}_X.npy".format(data_type_sel, k), np.asarray(data_container))
        np.save(dir_out + "{}_{}_y.npy".format(data_type_sel, k), np.asarray(label_container))


# Note that raw indicates raw 'lower' body data.

'''

/usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/opp/generate_data.py
------ raw data ------
(94260, 250)
test_101: (37284, 250)
test_102: (27199, 250)
test_104: (24220, 250)
test_105: (5557, 250)
test_data: (13464, 585)
test_label: (13464,)
(202579, 250)
train_101: (92752, 250)
train_102: (51823, 250)
train_104: (48121, 250)
train_105: (9883, 250)
train_data: (28938, 585)
train_label: (28938,)
(95275, 250)
valid_101: (41749, 250)
valid_102: (22518, 250)
valid_104: (26363, 250)
valid_105: (4645, 250)
valid_data: (13609, 585)
valid_label: (13609,)
------ upper data ------
(94260, 250)
test_101: (37284, 250)
test_102: (27199, 250)
test_104: (24220, 250)
test_105: (5557, 250)
test_data: (13464, 216)
test_label: (13464,)
(202579, 250)
train_101: (92752, 250)
train_102: (51823, 250)
train_104: (48121, 250)
train_105: (9883, 250)
train_data: (28938, 216)
train_label: (28938,)
(95275, 250)
valid_101: (41749, 250)
valid_102: (22518, 250)
valid_104: (26363, 250)
valid_105: (4645, 250)
valid_data: (13609, 216)
valid_label: (13609,)
------ lower data ------
(94260, 250)
test_101: (37284, 250)
test_102: (27199, 250)
test_104: (24220, 250)
test_105: (5557, 250)
test_data: (13464, 156)
test_label: (13464,)
(202579, 250)
train_101: (92752, 250)
train_102: (51823, 250)
train_104: (48121, 250)
train_105: (9883, 250)
train_data: (28938, 156)
train_label: (28938,)
(95275, 250)
valid_101: (41749, 250)
valid_102: (22518, 250)
valid_104: (26363, 250)
valid_105: (4645, 250)
valid_data: (13609, 156)
valid_label: (13609,)

Process finished with exit code 0


'''