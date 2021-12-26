import glob
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from dataset_analysis import readDataset, TRAINING_SET, VALIDATION_SET
import load_data
import parameters

DEBUG = True

# Loading pre-calculated statistics on the training data
TRAINING_SET_STAT = './data/train/stat.pickle'
with open(TRAINING_SET_STAT, 'rb') as f: stat = pickle.load(f)
TRAINING_STD = stat["std"]
TRAINING_MEAN = stat["mean"]
TRAINING_DATASIZE = stat["shape"][0]

# Loading pre-calculated statistics on the validation data
VALIDATION_SET_STAT = './data/validate/stat.pickle'
with open(VALIDATION_SET_STAT, 'rb') as f: stat = pickle.load(f)
VALIDATION_DATASIZE = stat["shape"][0]

def setup():
    # training = readDataset(TRAINING_SET)
    x_train, y_train, x_test = load_data.load_train_data(parameters.G_DataPath, forest=False)
    training, label = load_data.tran_learn(x_train, y_train)
    print("x_train.shape ->", training.shape)
    print("y_train.shape ->", label.shape)
    # validation = readDataset(VALIDATION_SET)
    validation = load_data.tran_validation(x_train, x_test)
    print("x_test.shape  ->", validation.shape)
    scaler = MinMaxScaler(feature_range=(-1, 1))  # range (-1, 1) is activation region of tanh (LSTM default act. func)
    scaler = scaler.fit(training)
    return training, validation, scaler

def parseDate(dateStr):
    """
    Replaces datetime.strptime() for performance gain
    """
    return datetime(
                int(dateStr[6:10]),  #year
                int(dateStr[3:5]),   #month
                int(dateStr[:2]),    #day
                int(dateStr[11:13]), #hour
                int(dateStr[14:16]), #minute
                int(dateStr[17:19]), #second
                )

def unstandardize(data):
    return data * TRAINING_STD + TRAINING_MEAN

def standardize(data):
    return (data - TRAINING_MEAN) / TRAINING_STD

if __name__ == '__main__':
    if DEBUG:
        print(f"Training std: {TRAINING_STD}, Training mean: {TRAINING_MEAN}, Training size: {TRAINING_DATASIZE}")
