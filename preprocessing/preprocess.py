import numpy as np
import nibabel as nib
import os
import glob
import yaml
import tqdm
import shutil


CHANNELS = ['flair', 't1', 't1ce', 't2']
SEGMENTATION = 'seg'


def read_instance(path, shape=(4, 240, 240, 155), load_y=False):
    """
    Read single instance of MRI image.
    Args:
        path (str): path to folder with images.
        shape (tuple(int)): shape of image to read.
        load_y (bool): flag whether to load target data.
    Returns:
        X (np.array) or (X, y) (tuple(np.array)): Instance of 4-channel 3D MRI image with
        optional ground truth segmentation.
    """
    X = np.empty(shape)
    for i, channel in enumerate(CHANNELS):
        channel_path = glob.glob(os.path.join(path, f'*{channel}.nii.gz'))[0]
        X[i] = nib.load(channel_path).get_fdata()

    if load_y:
        y_path = glob.glob(os.path.join(path, f'*{SEGMENTATION}.nii.gz'))[0]
        y = nib.load(y_path).get_fdata()
        return X, y

    return X


def create_output_dirs(path):
    """
    Create directory tree for processed data.
    Args:
        path (str): path to folder which stores processed data.
    Returns:
        train_path (str), val_path (str): Path to created directories.
    """

    def __create_output_dir(child_path):
        """Create part of dataset dirs structure"""
        if os.path.exists(child_path):
            shutil.rmtree(child_path)

        os.makedirs(os.path.join(child_path, 'X'))
        os.makedirs(os.path.join(child_path, 'y'))

    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    __create_output_dir(train_path)
    __create_output_dir(val_path)

    return train_path, val_path


def process_images_to_slices(input_path, output_path, threshold, val_size=0.2):
    """
    Read 3D MRI instances and save them as 2D slice
    Args:
        input_path (str): path to folder with training instances
        output_path (str): path to output folder
        threshold (float): background threshold to keep slice
        val_size (float): probability of slice to go to val set
    """

    def __save_slice(X, y, path, i):
        """Save single slice of image"""
        np.save(os.path.join(path, 'X', f'{i}.npy'), X.astype(np.float32))
        np.save(os.path.join(path, 'y', f'{i}.npy'), y.astype(np.uint8))

    np.random.seed(42)

    images_paths = glob.glob(os.path.join(input_path, '?GG', '*'))
    images_paths.sort()
    
    train_path, val_path = create_output_dirs(output_path)

    train_i, val_i = 0, 0
    for path in tqdm.tqdm(images_paths, desc='[INFO] Processing train/val images'):
        X, y = read_instance(path, load_y=True)
        mask = (y != 0).mean(axis=(0, 1)) > threshold
        for i, flag in enumerate(mask):
            if flag:
                if np.random.random() < val_size:  # save to validation
                    __save_slice(X[:, :, :, i], y[:, :, i], val_path, val_i)
                    val_i += 1
                else:  # save to train
                    __save_slice(X[:, :, :, i], y[:, :, i], train_path, train_i)
                    train_i += 1


def process_test(input_path, output_path):
    """
    Read 3D MRI test instances and save them as 3D volumes
    Args:
        input_path (str): path to folder with test instances
        output_path (str): path to output folder
    """
    images_paths = glob.glob(os.path.join(input_path, 'BraTS*'))
    images_paths.sort()
    
    test_path = os.path.join(output_path, 'test')
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.makedirs(test_path)

    for i, path in tqdm.tqdm(enumerate(images_paths), desc='[INFO] Processing test images'):
        X = read_instance(path, load_y=False)
        np.save(os.path.join(test_path, f'{i}.npy'), X.astype(np.float32))


if __name__ == '__main__':
    with open('config.yaml', "r") as config_file:
        config = yaml.full_load(config_file)

    process_images_to_slices(
        config['train_input_path'],
        config['preprocessed_path'],
        config['background_threshold'],
        config['validation_size']
    )

    process_test(
        config['test_input_path'],
        config['preprocessed_path'],
    )
