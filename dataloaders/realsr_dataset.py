import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from dataloaders.realesrgan import RealESRGAN_degradation

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(),
            ])

            self.gt_list = []
            assert len(args.dataset_txt_paths_list) == len(args.dataset_prob_paths_list)
            for idx_dataset in range(len(args.dataset_txt_paths_list)):
                with open(args.dataset_txt_paths_list[idx_dataset], 'r') as f:
                    dataset_list = [line.strip() for line in f.readlines()]
                    for idx_ratio in range(args.dataset_prob_paths_list[idx_dataset]):
                        gt_length = len(self.gt_list)
                        self.gt_list += dataset_list
                        print(f'=====> append {len(self.gt_list) - gt_length} data.')

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            # example["prompt"] = caption
            example["neg_prompt"] = self.args.neg_prompt
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t

            return example
        

class WaymoRepairDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop((512, 512)),
            transforms.RandomHorizontalFlip(),
        ])
        self.args = args
        self.split = split
        self.data_dir = args.data_dir
        self.gt_dir = os.path.join(self.data_dir, 'gt')
        self.lq_dir = os.path.join(self.data_dir, 'lq')
        self.gt_list = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        self.lq_list = sorted(glob.glob(os.path.join(self.lq_dir, '*.png')))
        assert len(self.gt_list) == len(self.lq_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.gt_list[idx]).convert('RGB')
        lq_img = Image.open(self.lq_list[idx]).convert('RGB')
        # to tensor
        gt_img = F.to_tensor(gt_img)
        lq_img = F.to_tensor(lq_img)
        assert gt_img.shape == lq_img.shape

        # 合并 gt，lq 特征
        img = torch.cat([gt_img, lq_img], dim=0)
        if img.shape[1] < 512 or img.shape[2] < 512:
            # 等比缩放
            scale_factor = 512 / min(img.shape[1], img.shape[2])
            img = F.resize(img, (int(img.shape[1] * scale_factor), 
                                 int(img.shape[2] * scale_factor)), 
                                 interpolation=transforms.InterpolationMode.BICUBIC)
        img = self.crop_preproc(img)
        gt_img, lq_img = img[:3, :, :], img[3:, :, :]
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])
        lq_img = F.normalize(lq_img, mean=[0.5], std=[0.5])

        example = {}
        example["neg_prompt"] = self.args.neg_prompt
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = lq_img
        return example

if __name__ == '__main__':
    dataset = WaymoRepairDataset(split='train', args=None)
    for i in range(10):
        example = dataset[i]
        print(example['conditioning_pixel_values'].shape)
        print(example['output_pixel_values'].shape)
    