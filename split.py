import numpy as np
import constants
from sklearn import model_selection
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

classes = np.load(constants.CLASSES_PATH)
classes -= 1
n_classes = len(np.unique(classes))
data = np.load(constants.DATA_PATH)
segments_id = np.load(constants.SEGMENT_ID_PATH)

assert len(classes) == len(data) == len(segments_id)


for i in range(constants.N_FOLDS):
    # permutation of data
    perm_data, perm_classes, perm_ids = shuffle(data, classes, segments_id, random_state=0)
    perm_classes = to_categorical(perm_classes, num_classes=n_classes)

    x_train, x_test_validation, y_train, y_test_validation, id_train, id_test_valid = model_selection.train_test_split(perm_data, perm_classes, perm_ids,
                                                                                              test_size=constants.TRAIN_SIZE,
                                                                                              random_state=42, stratify=perm_classes)
    x_validation, x_test, y_validation, y_test, id_valid, id_test = model_selection.train_test_split(x_test_validation,
                                                                                    y_test_validation, id_test_valid,
                                                                                    test_size=constants.TEST_SIZE, random_state=42, stratify=y_test_validation)


    """ SAVING DATA INTO FILES """
    out_path = constants.BASE_FOLDER_NAME + str(i+1) + "/"
    np.save(out_path + "x_train_"+str(i+1), x_train)
    np.save(out_path + "x_validation_"+str(i+1), x_validation)
    np.save(out_path + "x_test_"+str(i+1), x_test)
    np.save(out_path + "y_train_"+str(i+1), y_train)
    np.save(out_path + "y_validation_"+str(i+1), y_validation)
    np.save(out_path + "y_test_"+str(i+1), y_test)
    np.save(out_path + "id_train_"+str(i+1), id_train)
    np.save(out_path + "id_validation_"+str(i+1), id_valid)
    np.save(out_path + "id_test_"+str(i+1), id_test)



