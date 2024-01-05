from moments.train_hu_moments import *
from moments.test_hu_moments import *
from moments.train_zernike_moments import *
from moments.test_zernike_moments import *

def create_datasets(train_path, test_path, moment):
    train_dataframe, test_dataframe = None, None
    if moment == "zernike":
        train_dataframe = create_train_dataset_zernike(train_path)
        test_dataframe = create_test_dataset_zernike(test_path)
    elif moment == "hu":
        train_dataframe = create_train_dataset_hu(train_path)
        test_dataframe = create_test_dataset_hu(test_path)
    return train_dataframe, test_dataframe