import torchio as tio
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
from random import seed

def dataset_transform(reshape=(128, 128, 32), augumentation=False):
    list_of_transforms = [tio.ToCanonical(),
                          tio.Resize(target_shape=reshape),
                          tio.RescaleIntensity(out_min_max=(-1, 1))]
    if augumentation:
        list_of_transforms.append(tio.OneOf({tio.RandomAffine(degrees=15, translation=15): 1.,
                                             tio.RandomElasticDeformation(): 1.,
                                             tio.RandomNoise(mean=0.6): 1.,
                                             tio.RandomMotion(): 1.,
                                             tio.RandomGhosting(): 1.,
                                             tio.RandomSpike(): 1.,
                                             tio.RandomBiasField(): 1.,
                                             tio.RandomBlur(): 1.,
                                            })
                                 )
    return tio.Compose(list_of_transforms)
  

def split_dataset(dataset, test_size=0.2, train_transform=None, test_transform=None, seed=seed(), expand_ds=0):
    
    train_part, valid_part = train_test_split(dataset, test_size=0.2, random_state=seed)

    if train_transform is not None:
        train_dataset = tio.SubjectsDataset(train_part, transform=train_transform)
    else:
        train_dataset = tio.SubjectsDataset(train_part)
    
    for i in range(expand_ds):
        train_dataset = ConcatDataset([train_dataset, tio.SubjectsDataset(train_part, transform=train_transform)])
        
    if test_transform is not None:
        valid_dataset = tio.SubjectsDataset(valid_part, transform=test_transform)
    else:
        valid_dataset = tio.SubjectsDataset(valid_part)
    
    return train_dataset, valid_dataset
