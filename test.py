
from data_loader import file5array,file5ori_array,file5ori_array_1
from model import get_model

from keras.models import load_model

import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

seed = 7
import pandas as pd
import numpy
numpy.random.seed(seed)

model = get_model()
print("loading model...")
# Compile
adam = tf.keras.optimizers.Adam(lr=0.000025, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.3)

def get_lr_metric(optimizer):

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

lr_metric = get_lr_metric(adam)

model = load_model('fei_model.h5',custom_objects={'lr':lr_metric})
print("start predicting...")

x_test = file5ori_array("./test")
result=model.predict(x_test)
#res = model.predict(xx)

df = pd.DataFrame(data=result)
database = pd.read_csv('sampleSubmission.csv')
database['Score'] = df
database.rename(columns = {database.columns[0]:'ID'}, inplace=True)
database.rename(columns = {database.columns[1]:'Predicted'}, inplace=True)
database.to_csv("submission.csv", index=False)

print("finished!")
#https://blog.csdn.net/lengxiaomo123/article/details/68926778/
#https://blog.csdn.net/C_chuxin/article/details/83422454

