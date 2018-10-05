import numpy as np
import matplotlib.pyplot as plt
import gin
from tqdm import tqdm
import SimpleITK as sitk

@gin.configurable
def dicom_to_array(dicom_files, verbose=True):
    assert isinstence(dicom_files, list), 'Dicom files must be sent as a list, not {}'.format(type(dicom_files))
    for file in tqdm(dicom_files):
        itkimage = sitk.ReadImage(file)
        array = sitk.GetArrayFromImage(itkimage)
        array = np.fliplr(np.rot90(np.swapaxes(array,0,2),-1))
        array = (array - np.mean(array)) / np.std(array)
        np.save(file, array)
        if verbose:
            print(f'DICOM file {file} saved to Numpy format.')
        #plt.imshow(array[:,:,0], cmap="bone")
        #plt.show()
