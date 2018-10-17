import os
import pandas as pd
import gin
import shutil
from sklearn.model_selection import train_test_split

@gin.configurable
class FilePrep:
    def __init__(self, project_dir=None, remake_dirs=False, train=0.60):
        self.project_dir = project_dir
        self.proj_image_dir = os.path.join(project_dir, 'images')
        self.remake_dirs = remake_dirs
        self.train = train

    def _make_class_df(self, label_df_name='stage_1_detailed_class_info.csv', base_data_dir=None, file_type='png',
                       file_name_col='patientId', image_paths_col='image_paths'):
        labels_df_path = os.path.join(self.project_dir, label_df_name)
        labels_df = pd.read_csv(labels_df_path)
        file_list = []
        file_names = []
        file_type = '.' + file_type
        for root, _, files in os.walk(base_data_dir):
            for file in files:
                if file.endswith(file_type):
                    file_names.append(file.replace(file_type, ''))
                    file_list.append(os.path.join(root, file))
        files_df = pd.DataFrame.from_dict({file_name_col: file_names, image_paths_col:file_list})
        data = pd.merge(files_df, labels_df, how='inner', on=file_name_col)
        data = data.drop_duplicates(subset=[file_name_col, image_paths_col])
        data_no_id = data.drop(columns=[file_name_col])
        #path_class = data_no_id.to_csv('path_class.csv', index=False)
        return data_no_id

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

    def unique_label_names(self, labels):
            return labels.unique()

    def _train_test_val(self, data_labels_df, image_paths_col=None, label_col=None):
        img_train, img_test_val, label_train, label_test_val  = train_test_split(data_labels_df[image_paths_col],
                                                                                 data_labels_df[label_col],
                                                                                 train_size=self.train,
                                                                                 test_size=1-self.train,
                                                                                 random_state=42)
        img_test, img_val, label_test, label_val = train_test_split(img_test_val, label_test_val,
                                                                    train_size=0.5, test_size=1 - self.train,
                                                                    random_state=42)
        data_dict = {'img_train':img_train,'label_train':label_train,'img_test':img_test,'label_test':label_test,
                     'img_val':img_val,'label_val':label_val}
        return data_dict

    @gin.configurable
    def build_dataset(self, label_df_name='stage_1_detailed_class_info.csv', file_type='png', file_name_col='patientId',
                      label_col=None, image_paths_col='image_paths', base_data_dir='/data/gferguso/cord_comp/Current_data_subset'):
        data_labels = self._make_class_df(label_df_name=label_df_name, file_type=file_type, file_name_col=file_name_col,
                                          image_paths_col=image_paths_col,base_data_dir=base_data_dir)
        data_labels.to_csv(os.path.join(self.proj_image_dir, 'names_labels'))
        train_base = os.path.join(self.proj_image_dir , 'train')
        test_base  = os.path.join(self.proj_image_dir , 'test')
        valid_base = os.path.join(self.proj_image_dir , 'validate')
        label_names = self.unique_label_names(data_labels[label_col])
        data_dict = self._train_test_val(data_labels, image_paths_col=image_paths_col, label_col=label_col)
        for i, label in enumerate(label_names):
            s_label = label.strip().replace('/', 'and').replace(' ', '_')
            dir_name = f'class_{i:03d}_{s_label}'
            train_path = os.path.join(train_base, dir_name)
            test_path  = os.path.join(test_base, dir_name)
            valid_path = os.path.join(valid_base, dir_name)
            path_classes = [train_path, test_path, valid_path]
            self._make_dirs(path_classes)
            train_files =  data_dict['img_train'][data_dict['label_train'] == label].tolist()
            for file in train_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(train_path, f_name)
                if not os.path.exists(file_dest):
                    shutil.copy(file, file_dest)
            test_files =  data_dict['img_test'][data_dict['label_test'] == label].tolist()
            for file in test_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(test_path, f_name)
                if not os.path.exists(file_dest):
                    shutil.copy(file, file_dest)
            valid_files =  data_dict['img_val'][data_dict['label_val'] == label].tolist()
            for file in valid_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(valid_path, f_name)
                if not os.path.exists(file_dest):
                    shutil.copy(file, file_dest)
        print('Directory structure built and train/test/validation files moved to directories.')
