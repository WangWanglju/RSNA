import cv2
import torch
from torch.utils.data import Dataset
from config import CFG
class TrainDataset(Dataset):
    """
    Image torch Dataset.
    """
    def __init__(
        self,
        cfg,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            paths (list): Path to images.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.config = cfg
        self.paths = df['path'].values
        self.transforms = transforms
        self.targets = df['cancer'].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.paths[idx])
        if self.transforms:
            image = self.transforms(image=image)["image"]

        y = torch.tensor([self.targets[idx]], dtype=torch.float)
        # w = torch.tensor([1])

        return image, y


import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_transfos(augment=True, visualize=False):
    """
    Returns transformations.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.Compose(
        [   albu.Resize(512, 512),
            albu.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        p=1,
    )


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    SAVE_FOLDER = '../input/rsna-breast-cancer-detection/train_images/'
    
    path = '../input/rsna-breast-cancer-detection/train.csv'

    train = pd.read_csv(path)
    train['path'] = SAVE_FOLDER + train["patient_id"].astype(str) + "_" + train["image_id"].astype(str) + ".png"
    train_dataset = TrainDataset(CFG, train, get_transfos())

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.data_config['batch_size'],
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
    for i,j in train_loader:
        print(i,i)
        break

                            