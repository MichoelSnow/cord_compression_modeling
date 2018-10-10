import os
import pandas as pd
import gin
import shutil
from sklearn.model_selection import train_test_split

@gin.configurable
class FilePrep:
    def __init__(self, img_dataframe=None, remake_dirs=False, label_col=None, train=0.60):
        self.data_dir = os.path.join(os.getcwd(), 'images')
        assert isinstance(img_dataframe, pd.DataFrame), f'{img_dataframe} must be a Dataframe.'
        self.data = img_dataframe.drop(columns=label_col)
        self.labels = img_dataframe[label_col]
        self.remake_dirs = remake_dirs
        self.train = train

    def _make_dirs(self, paths):
        assert isinstance(paths, list), 'Only make directories from lists, not {}'.format(type(paths))
        for path in paths:
            if self.remake_dirs:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path)
                else:
                    os.makedirs(path)
            else:
                try:
                    os.makedirs(path)
                except OSError:
                    print(f'Directory {path} already exists. To remake set remake_dirs=True.')

    def label_names(self):
            return self.labels.unique()

    def _train_test_val(self):
        img_train, img_test_val, label_train, label_test_val  = train_test_split(self.data, self.labels,
                                                                                 train_size=self.train,
                                                                                 test_size=1-self.train,
                                                                                 random_state=42)
        img_test, img_val, label_test, label_val = train_test_split(img_test_val, label_test_val,
                                                                    train_size=0.5, test_size=1 - self.train,
                                                                    random_state=42)
        data_dict = {'img_train':img_train,'label_train':label_train,'img_test':img_test,'label_test':label_test,
                     'img_val':img_val,'label_val':label_val}
        return data_dict

    def build_dataset(self, data_dir=None):
        if not data_dir:
            data_dir=self.data_dir
        train_base = os.path.join(data_dir, 'train')
        test_base  = os.path.join(data_dir, 'test')
        valid_base = os.path.join(data_dir, 'validate')
        label_names = self.label_names()
        data_dict = self._train_test_val()
        for i, label in enumerate(label_names):
            s_label = label.strip().replace('/', 'and').replace(' ', '_')
            dir_name = f'class_{i:03d}_{s_label}'
            train_path = os.path.join(train_base, dir_name)
            test_path  = os.path.join(test_base, dir_name)
            valid_path = os.path.join(valid_base, dir_name)
            path_classes = [train_path, test_path, valid_path]
            self._make_dirs(path_classes)
            train_files =  data_dict['img_train'][data_dict['label_train'] == label]['image_paths'].tolist()
            print(train_files)
            for file in train_files:
                if not os.path.exists(file):
                    _, f_name = os.path.split(file)
                    file_dest = os.path.join(train_path, f_name)
                    shutil.copy(file, file_dest)
            test_files =  data_dict['img_test'][data_dict['label_test'] == label]['image_paths'].tolist()
            for file in test_files:
                if not os.path.exists(file):
                    _, f_name = os.path.split(file)
                    file_dest = os.path.join(test_path, f_name)
                    shutil.copy(file, file_dest)
            valid_files =  data_dict['img_val'][data_dict['label_val'] == label]['image_paths'].tolist()
            for file in valid_files:
                if not os.path.exists(file):
                    _, f_name = os.path.split(file)
                    file_dest = os.path.join(valid_path, f_name)
                    shutil.copy(file, file_dest)
        print('Directory structure built and train/test/validation files moved to directories.')
