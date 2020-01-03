import numpy as np

from data_loader import file5array,file5ori_array,file5ori_array_1
from model import get_model


import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

seed = 7
import numpy
numpy.random.seed(seed)
import pandas as pd



tmp = np.load('./train_val/candidate1.npz')

model = get_model()
# 160*100*22
'''
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr
'''




# Compile
adam = tf.keras.optimizers.Adam(lr=0.000025, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.3)
#lr_metric = get_lr_metric(adam)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', lr_metric])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_data  = file5array("./train_val",20,'train')
test_data = file5array("./train_val",20,'test')
#X = train_data[0]
#Y = train_data [1]
#print(file5array("./train_val",1))
x_test = file5ori_array("./test")

#print(X,Y)
#X_train, X_test, y_train, y_test = train_test_split(train_data, test_size=0.33, random_state=seed)
print("start fitting")
earlyStopping= tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
model.fit_generator(train_data, validation_data=test_data, validation_steps=5, steps_per_epoch=18,
                    epochs=9,callbacks=[earlyStopping])
#model = load_model('fei_model.h5')

print("finished fitting")
model.save('fei_model_nono.h5')

#score = model.evaluate(x_test, y_test, batch_size=32)

result=model.predict(x_test)

print('shape of result',result.shape)

df = pd.DataFrame(data=result)
database = pd.read_csv('sampleSubmission.csv')
database['Score'] = df
database.rename(columns = {database.columns[0]:'ID'}, inplace=True)
database.rename(columns = {database.columns[1]:'Predicted'}, inplace=True)
database.to_csv("data_final.csv", index=False)

'''
print('shape of res',res.shape)

for k in range (0, 465):
    i = float(res[k])
    print(i)
'''

#https://blog.csdn.net/lengxiaomo123/article/details/68926778/
#https://blog.csdn.net/C_chuxin/article/details/83422454

