#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from curses import meta
from gc import callbacks
from random import shuffle
from termios import VLNEXT
from helper_code import *
from team_code import base_model, load_challenge_model, build_murmur_model, build_clinical_model, scheduler, scheduler_2, get_murmur_locations, pad_array, calculating_class_weights
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import os
import tqdm
import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################



# Cross validate model.
def cv_challenge_model(data_folder, result_folder, n_epochs_1, n_epochs_2, n_folds, pre_train):
    NEW_FREQUENCY = 100
    batch_size = 30

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(result_folder, exist_ok=True)
    #TODO: remove this:
    #classes = ['Present', 'Unknown', 'Absent']
    #num_classes = len(classes)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    #Make CV folds
    cv_murmur = []
    cv_outcome = []
    max_len = 0
    for i in tqdm.tqdm(range(len(patient_files))):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        cv_murmur.append(get_murmur(current_patient_data))
        cv_outcome.append(get_outcome(current_patient_data))
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        for indx, rec in enumerate(current_recordings):
            rec_len = int(len(rec)/freq[indx]*NEW_FREQUENCY)
            if rec_len > max_len:
                max_len = rec_len
    cv_outcome = np.asarray(cv_outcome)
    cv_murmur = np.asarray(cv_murmur)
    patient_files = np.asarray(patient_files)
 
    skf = StratifiedKFold(n_splits=n_folds)

    murmur_probas = []
    outcome_probas = []
    murmur_trues = []
    outcome_trues = []
    clinical_history = []
    murmur_history = []
    patient_labels = []
    val_murmur_patient_clf_cv = []
    val_outcome_patient_clf_cv = []

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2, verbose=0)

    for train_index, val_index in skf.split(cv_murmur, cv_outcome):
        train_data, train_murmurs, train_outcomes, _ = get_data(patient_files[train_index], data_folder, NEW_FREQUENCY, num_murmur_classes, num_outcome_classes,outcome_classes,max_len)
        print(f"Number of signals in training data = {train_data.shape[0]}")
        print("Murmurs prevalence:")
        print(f"Present = {np.where(np.argmax(train_murmurs,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(train_murmurs,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(train_murmurs,axis=1)==2)[0].shape[0]}")
        print("Outcomes prevalence:")
        print(f"Abnormal = {len(np.where(train_outcomes==0)[0])}, Normal = {len(np.where(train_outcomes==1)[0])}")
        
        
        val_data, val_murmurs, val_outcomes, val_patient_labels= get_data(patient_files[val_index], data_folder, NEW_FREQUENCY, num_murmur_classes, num_outcome_classes,outcome_classes, max_len)
        print(f"Number of signals in validation data = {val_data.shape[0]}")
        print("Murmurs prevalence:")
        print(f"Present = {np.where(np.argmax(val_murmurs,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(val_murmurs,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(val_murmurs,axis=1)==2)[0].shape[0]}")
        print("Outcomes prevalence:")
        print(f"Abnormal = {len(np.where(val_outcomes==0)[0])}, Normal = {len(np.where(val_outcomes==1)[0])}")
        val_murmur_patient_clf = cv_murmur[val_index]
        val_outcome_patient_clf = cv_outcome[val_index]

        # Calculate weights
        new_weights_murmur=calculating_class_weights(train_murmurs)
        keys = np.arange(0,len(murmur_classes),1)
        murmur_weight_dictionary = dict(zip(keys, new_weights_murmur.T[1]))

        weight_outcome = np.unique(train_outcomes, return_counts=True)[1][0]/np.unique(train_outcomes, return_counts=True)[1][1]
        outcome_weight_dictionary = {0: 1.0, 1:weight_outcome}

        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)

        with strategy.scope():
            if pre_train == False:
                # Initiate the model.
                clinical_model = build_clinical_model(train_data.shape[1],train_data.shape[2])
                murmur_model = build_murmur_model(train_data.shape[1],train_data.shape[2])
                print("Train murmur model..")
                temp_murmur_history = murmur_model.fit(x=train_data, y=train_murmurs, epochs=n_epochs_1, batch_size=batch_size,   
                        verbose=1, validation_data = (val_data,val_murmurs),
                        class_weight=murmur_weight_dictionary, shuffle = True,
                        callbacks=[lr_schedule]
                        )

                print("Train clinical model..")
                temp_clinical_history = clinical_model.fit(x=train_data, y=train_outcomes, epochs=n_epochs_2, batch_size=batch_size,  
                        verbose=1, validation_data = (val_data,val_outcomes),
                        class_weight=outcome_weight_dictionary, shuffle = True,
                        callbacks=[lr_schedule]
                        )
            elif pre_train == True:
                print("Train murmur model..")
                model = base_model(train_data.shape[1],train_data.shape[2])
                model.load_weights("./pretrained_model.h5")
                
                murmur_layer = tf.keras.layers.Dense(3, "softmax",  name="murmur_output")(model.layers[-2].output)
                murmur_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[murmur_layer])
                
                for layer in murmur_model.layers[:-2]:
                    layer.trainable = False

                murmur_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='ROC')])
                
                murmur_model.fit(x=train_data, y=train_murmurs, epochs=5, batch_size=batch_size,   
                        verbose=1, validation_data = (val_data,val_murmurs),
                        class_weight=murmur_weight_dictionary, shuffle = True)
                
                for layer in murmur_model.layers[:-2]:
                    layer.trainable = True

                murmur_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='ROC')])
                
                temp_murmur_history = murmur_model.fit(x=train_data, y=train_murmurs, epochs=n_epochs_1, batch_size=batch_size,   
                        verbose=1, validation_data = (val_data,val_murmurs),
                        class_weight=murmur_weight_dictionary, shuffle = True, callbacks=[lr_schedule])
                
                print("Train clinical model..")
                outcome_layer = tf.keras.layers.Dense(1, "sigmoid",  name="clinical_output")(model.layers[-2].output)
                clinical_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[outcome_layer])
                for layer in clinical_model.layers[:-2]:
                    layer.trainable = False
                clinical_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC')])
                
                clinical_model.fit(x=train_data, y=train_outcomes, epochs=5, batch_size=batch_size,  
                        verbose=1, validation_data = (val_data,val_outcomes),
                        class_weight=outcome_weight_dictionary, shuffle = True)
                
                for layer in clinical_model.layers[:-2]:
                    layer.trainable = True
                
                clinical_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC')])
                
                temp_clinical_history = clinical_model.fit(x=train_data, y=train_outcomes, epochs=n_epochs_2, batch_size=batch_size,  
                        verbose=1, validation_data = (val_data,val_outcomes),
                        class_weight=outcome_weight_dictionary, shuffle = True, callbacks=[lr_schedule])

            murmur_probabilities = murmur_model.predict(val_data)

            outcome_probabilities = clinical_model.predict(val_data)

        clinical_history.append(temp_clinical_history)
        murmur_history.append(temp_murmur_history)
        murmur_probas.append(murmur_probabilities)
        outcome_probas.append(outcome_probabilities)
        murmur_trues.append(val_murmurs)
        outcome_trues.append(val_outcomes)
        patient_labels.append(val_patient_labels)
        val_murmur_patient_clf_cv.append(val_murmur_patient_clf)
        val_outcome_patient_clf_cv.append(val_outcome_patient_clf)

    return murmur_model, clinical_model, murmur_probas, outcome_probas, murmur_trues, outcome_trues, murmur_history, clinical_history, val_data, patient_labels, val_murmur_patient_clf_cv, val_outcome_patient_clf_cv


def get_data(patient_files, data_folder, new_frequenzy, num_murmur_classes, num_outcome_classes,outcome_classes, max_length):
    data = []
    murmurs = list()
    outcomes = list()
    patient_label = list()
    for i in tqdm.tqdm(range(len(patient_files))):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        for j in range(len(current_recordings)):
            data.append(signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * new_frequenzy)))
            current_auscultation_location = current_patient_data.split('\n')[1:len(current_recordings)+1][j].split(" ")[0]
            all_murmur_locations = get_murmur_locations(current_patient_data).split("+")
            current_murmur = np.zeros(num_murmur_classes, dtype=int)
            if get_murmur(current_patient_data) == "Present":
                if current_auscultation_location in all_murmur_locations:
                    current_murmur[0] = 1
                else:
                    pass
            elif get_murmur(current_patient_data) == "Unknown":
                current_murmur[1] = 1
            elif get_murmur(current_patient_data) == "Absent":
                current_murmur[2] = 1
            murmurs.append(current_murmur)

            current_outcome = np.zeros(num_outcome_classes, dtype=int)
            outcome = get_outcome(current_patient_data)
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
            outcomes.append(current_outcome)
            patient_label.append(i)
    data_padded = pad_array(data, max_length)
    data_padded = np.expand_dims(data_padded,2)

    murmurs = np.vstack(murmurs)
    outcomes = np.argmax(np.vstack(outcomes),axis=1)
    patient_label = np.asarray(patient_label)

    return data_padded, murmurs, outcomes, patient_label
