#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
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




# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    PRE_TRAIN = False
    NEW_FREQUENCY = 100 # longest signal, while resampling to 500Hz = 32256 samples
    EPOCHS_1 = 30
    EPOCHS_2 = 20
    BATCH_SIZE_1 = 20
    BATCH_SIZE_2 = 20

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    #TODO: remove this:
    #classes = ['Present', 'Unknown', 'Absent']
    #num_classes = len(classes)

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    data = []
    murmurs = list()
    outcomes = list()
    

    for i in tqdm.tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, freq = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        for j in range(len(current_recordings)):
            data.append(signal.resample(current_recordings[j], int((len(current_recordings[j])/freq[j]) * NEW_FREQUENCY)))
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

    data_padded = pad_array(data)
    data_padded = np.expand_dims(data_padded,2)

    murmurs = np.vstack(murmurs)
    outcomes = np.argmax(np.vstack(outcomes),axis=1)
    print(f"Number of signals = {data_padded.shape[0]}")

    # The prevalence of the 3 different labels
    print("Murmurs prevalence:")
    print(f"Present = {np.where(np.argmax(murmurs,axis=1)==0)[0].shape[0]}, Unknown = {np.where(np.argmax(murmurs,axis=1)==1)[0].shape[0]}, Absent = {np.where(np.argmax(murmurs,axis=1)==2)[0].shape[0]}")

    print("Outcomes prevalence:")
    print(f"Abnormal = {len(np.where(outcomes==0)[0])}, Normal = {len(np.where(outcomes==1)[0])}")

    new_weights_murmur=calculating_class_weights(murmurs)
    keys = np.arange(0,len(murmur_classes),1)
    murmur_weight_dictionary = dict(zip(keys, new_weights_murmur.T[1]))
  
    weight_outcome = np.unique(outcomes, return_counts=True)[1][0]/np.unique(outcomes, return_counts=True)[1][1]
    outcome_weight_dictionary = {0: 1.0, 1:weight_outcome}

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2, verbose=0)

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        if PRE_TRAIN == False:
            # Initiate the model.
            clinical_model = build_clinical_model(data_padded.shape[1],data_padded.shape[2])
            murmur_model = build_murmur_model(data_padded.shape[1],data_padded.shape[2])
        elif PRE_TRAIN == True:
            model = base_model(data_padded.shape[1],data_padded.shape[2])
            model.load_weights("./pretrained_model.h5")
            
            outcome_layer = tf.keras.layers.Dense(1, "sigmoid",  name="clinical_output")(model.layers[-2].output)
            clinical_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[outcome_layer])
            clinical_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                metrics = [tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(curve='ROC')])

            murmur_layer = tf.keras.layers.Dense(3, "softmax",  name="murmur_output")(model.layers[-2].output)
            murmur_model = tf.keras.Model(inputs=model.layers[0].output, outputs=[murmur_layer])
            murmur_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC(curve='ROC')])
        


        murmur_model.fit(x=data_padded, y=murmurs, epochs=EPOCHS_1, batch_size=BATCH_SIZE_1,   
                    verbose=1, shuffle = True,
                    class_weight=murmur_weight_dictionary
                    #,callbacks=[lr_schedule]
                    )

        clinical_model.fit(x=data_padded, y=outcomes, epochs=EPOCHS_2, batch_size=BATCH_SIZE_2,   
                    verbose=1, shuffle = True,
                    class_weight=outcome_weight_dictionary
                    #,callbacks=[lr_schedule]
                    )
    
    murmur_model.save(os.path.join(model_folder, 'murmur_model.h5'))

    clinical_model.save(os.path.join(model_folder, 'clinical_model.h5'))

    # Save the model.
    #save_challenge_model(model_folder, classes, imputer, classifier)



# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    model_dict = {}
    for i in os.listdir(model_folder):
        model = tf.keras.models.load_model(os.path.join(model_folder, i))
        model_dict[i.split(".")[0]] = model    
    return model_dict


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    NEW_FREQUENCY = 100

    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    # Load the data.
    #indx = get_lead_index(data)
    #extracted_recordings = np.asarray(recordings)[indx]
    new_sig_len = model["murmur_model"].get_config()['layers'][0]['config']['batch_input_shape'][1]
    data_padded = np.zeros((len(recordings),int(new_sig_len),1))
    freq = get_frequency(data)
    murmur_probabilities_temp = np.zeros((len(recordings),3))
    outcome_probabilities_temp = np.zeros((len(recordings),1))

    for i in range(len(recordings)):
        data = np.zeros((1,new_sig_len,1))
        rec = np.asarray(recordings[i])
        resamp_sig = signal.resample(rec, int((len(rec)/freq) * NEW_FREQUENCY))
        data[0,:len(resamp_sig),0] = resamp_sig
        
        murmur_probabilities_temp[i,:] = model["murmur_model"].predict(data)
        outcome_probabilities_temp[i,:] = model["clinical_model"].predict(data)

    avg_outcome_probabilities = np.sum(outcome_probabilities_temp)/len(recordings)
    avg_murmur_probabilities = np.sum(murmur_probabilities_temp,axis = 0)/len(recordings)

    binarized_murmur_probabilities = np.argmax(murmur_probabilities_temp, axis = 1)
    binarized_outcome_probabilities = (outcome_probabilities_temp > 0.5) * 1

    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    #murmur_indx = np.bincount(binarized_murmur_probabilities).argmax()
    #murmur_labels[murmur_indx] = 1
    if 0 in binarized_murmur_probabilities:
        murmur_labels[0] = 1
    elif 2 in binarized_murmur_probabilities:
        murmur_labels[2] = 1
    elif 1 in binarized_murmur_probabilities:
        murmur_labels[1] = 1

    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    # 0 = abnormal outcome
    if 0 in binarized_outcome_probabilities:
        outcome_labels[0] = 1
    else:
        outcome_labels[1] = 1

                                                        
    outcome_probabilities = np.array([avg_outcome_probabilities,1-avg_outcome_probabilities])
    murmur_probabilities = avg_murmur_probabilities


    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities.ravel(), outcome_probabilities.ravel()))
    
    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)

def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def base_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    return model

def build_murmur_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    murmur_output = tf.keras.layers.Dense(3, activation='softmax', name="murmur_output")(gap_layer)
    #clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=murmur_output)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = [tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.AUC(curve='ROC')])
    return model

def build_clinical_model(sig_len,n_features, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(shape=(sig_len,n_features))

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    clinical_output = tf.keras.layers.Dense(1, activation='sigmoid', name="clinical_output")(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=clinical_output)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = [tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.AUC(curve='ROC')])
    
    return model


def get_lead_index(patient_metadata):    
    lead_name = []
    lead_num = []
    cnt = 0
    for i in patient_metadata.splitlines(): 
        if i.split(" ")[0] == "AV" or i.split(" ")[0] == "PV" or i.split(" ")[0] == "TV" or i.split(" ")[0] == "MV":
            if not i.split(" ")[0] in lead_name:
                lead_name.append(i.split(" ")[0])
                lead_num.append(cnt)
            cnt += 1
    return np.asarray(lead_num)

def scheduler(epoch, lr):
    if epoch == 10:
        return lr * 0.1
    elif epoch == 15:
        return lr * 0.1
    elif epoch == 20:
        return lr * 0.1
    else:
        return lr

def scheduler_2(epoch, lr):
    return lr - (lr * 0.1)

def get_murmur_locations(data):
    murmur_location = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                murmur_location = l.split(': ')[1]
            except:
                pass
    if murmur_location is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return murmur_location

def pad_array(data, signal_length = None):
    max_len = 0
    for i in data:
        if len(i) > max_len:
            max_len = len(i)
    if not signal_length == None:
        max_len = signal_length
    new_arr = np.zeros((len(data),max_len))
    for j in range(len(data)):
        new_arr[j,:len(data[j])] = data[j]
    return new_arr

def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight(class_weight='balanced', classes=[0.,1.], y=y_true[:, i])
    return weights