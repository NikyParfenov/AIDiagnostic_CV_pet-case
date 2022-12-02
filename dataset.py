import copy
import torchio as tio
from sys import platform
from glob import glob
from random import randint
from torchio import Subject, SubjectsDataset
from typing import Callable, Iterable, List, Optional, Sequence


class ImageSubjectsDataset(SubjectsDataset):
    # For dataset let's use SubjectDataset from torchio (lib for 3D images developed in 2021). 
    # The main different between Dataset from torch is simplier load medical data.
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 transform: Optional[Callable] = None,
                 load_getitem: bool = True,
                ):
        self.paths = {'imgs': img_dir,
                      'masks': mask_dir}
        self._subjects = self._get_subjects()
        self._transform: Optional[Callable]
        self.set_transform(transform)
        self.load_getitem = load_getitem
        self._parse_subjects_list(self._subjects)
        
    def _get_item_list(self) -> List[str]:
        """
        Method get the only folders that contain both images and masks
        """
        folders = []
        for key in self.paths.keys():
            folder_path = glob(f'{self.paths[key]}/*')
            if platform == 'win32':
                folders.append(list(map(lambda x: x.split('\\')[-1], folder_path)))
            else:
                folders.append(list(map(lambda x: x.split('/')[-1], folder_path)))
        return list(set(folders[0]) & set(folders[1]))
    
    def _get_subjects(self) -> Sequence[Subject]:
        """
    	Method collect images and masks to list with Subjects
    	"""
        dataset_subjects = []
        for folder_name in self._get_item_list():
            img_folder = glob(f'{self.paths["imgs"]}/{folder_name}/**/', recursive=True)[-1]
            mask_file = glob(f'{self.paths["masks"]}/{folder_name}/*.nii.gz')[0] 
            dataset_subject = tio.Subject(image = tio.ScalarImage(img_folder),
                                          segmentation = tio.LabelMap(mask_file),
                                         )
            dataset_subjects.append(dataset_subject)
        return dataset_subjects
    
    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, index: int) -> Subject:
        try:
            index = int(index)
        except (RuntimeError, TypeError):
            message = (
                f'Index "{index}" must be int or compatible dtype,'
                f' but an object of type "{type(index)}" was passed'
            )
            raise ValueError(message)

        subject = self._subjects[index]
        subject = copy.deepcopy(subject)
        if self.load_getitem:
            subject.load()

        if self._transform is not None:
            subject = self._transform(subject)
        return subject
    

# from torch.utils.data import Dataset
# import nibabel as nib
# import SimpleITK as sitk

# class ImageDataset(Dataset):
#     def __init__(self, mask_dir, img_dir, transform=None, target_transform=None):
#         self.transform = transform
#         self.target_transform = target_transform
#         self.paths = {'imgs': img_dir,
#                       'masks': mask_dir}

#     def __len__(self):
#         return len(self._get_item_list())

#     def __getitem__(self, item=None):
#         if item is None:
#             folder_name = self._get_item_list()[randint(0, self.__len__()-1)]
#         else:
#             folder_name = self._get_item_list()[item]
#         img_folder = glob.glob(f'{self.paths["imgs"]}/{folder_name}/**/', recursive=True)[-1]
#         image = self._load_dicom(img_folder)
        
#         mask_file = glob.glob(f'{self.paths["masks"]}/{folder_name}/*')[0]
#         mask = nib.load(mask_file)
#         mask = mask.get_fdata().transpose(2, 1, 0)   
        
#         if self.transform:
#             # reshape for torchio transform
#             image = image.transpose(1, 2, 0)[None, :, :, :]
#             image = self.transform(image)
#         if self.target_transform:
#             # reshape for torchio transform
#             mask = mask.transpose(1, 2, 0)[None, :, :, :]
#             mask = self.target_transform(mask)
#         return image, mask
    
#     def _get_item_list(self):
#         """
#         Get the only folders that contain both images and masks
#         """
#         folders = []
#         for key in self.paths.keys():
#             folder_path = glob.glob(f'{self.paths[key]}/*')
#             folders.append(list(map(lambda x: x.split('/')[-1], folder_path)))
#         return list(set(folders[0]) & set(folders[1]))
    
#     def _load_dicom(self, directory):
#         reader = sitk.ImageSeriesReader()
#         dicom_names = reader.GetGDCMSeriesFileNames(directory)
#         reader.SetFileNames(dicom_names)
#         image_itk = reader.Execute()
#         image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
#         return image_zyx 
