# SJTU_2019_machine_learning_project
It is a project in SJTU machine learning course.
# Code Structure

* [`test/`](test/): contains 117 files of test
* [`train_val/`](train_val/): contains 465 files of train
* [`dataloader.py`](dataloader.py):a script of loading data.
* [`fei_model.h5`](fei_model.h5):a trained model file.
* [`model.py`](model.py): the model script.
* [`sampleSubmission.csv`](sampleSubmission.csv): the training script.
* [`test.py`](test.py): load the trained model "fei_model.h5" and predict the output of test,after running this script, a file named"submission.csv"can be created.
* [`train.py`](train.py): the training script.
* [`train_val.csv`](train_val.csv): the label of files in [`train_val/`](train_val/)
# How to run this project
1.Run train.py, then the train will begin, after traning, a file "data_final.csv" will be created. This file is the predicted of test data.
another file named "fei_model_nono.h5" will be created, too. That file is the saved model of training.

2.Run test.py, then model "fei_model.h5" will be loaded. This script will use the loaded model to predict the output of test data.
  After predicting, a file "submission.csv" will be created, which is the predicted result of test data.
