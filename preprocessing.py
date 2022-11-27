import torchio as tio
from sklearn.model_selection import train_test_split
from random import seed


def dataset_transform(reshape=(128, 128, 32), augumentation=False):
    list_of_transforms = [tio.ToCanonical(),
                          tio.Resize(target_shape=reshape),
                          tio.RescaleIntensity(out_min_max=(-1, 1))]
    if augumentation:
        list_of_transforms.append(tio.OneOf({tio.RandomAffine(degrees=15, translation=15): 0.5,
                                             tio.RandomElasticDeformation(): 0.2,
                                             tio.RandomFlip(): 0.3,
                                             tio.RandomNoise(mean=0.6): 0.2,
                                            })
                                 )
    return tio.Compose(list_of_transforms)
  
  
def split_dataset(dataset, test_size=0.2, train_transform=None, test_transform=None, seed=seed()):
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=seed)

    if (train_transform is None) and (test_transform is None):
	    return tio.SubjectsDataset(train_dataset), tio.SubjectsDataset(valid_dataset)
	    
    if train_transform is not None:
        train_dataset = tio.SubjectsDataset(train_dataset, transform=train_transform)
    if test_transform is not None:
        valid_dataset = tio.SubjectsDataset(valid_dataset, transform=test_transform)
    
    return train_dataset, valid_dataset
    
