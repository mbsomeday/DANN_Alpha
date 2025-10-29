import numpy as np
import os, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from configs.ds_path import PATHS


class my_dataset(Dataset):
    def __init__(self, ds_name_list, path_key, txt_name, ds_labels=None):
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key

        # 用于测试打乱dataset name和label的对应实验
        if ds_labels is None:
            for ds_name in ds_name_list:
                self.ds_label_list.append(int(ds_name[1]) - 1)
            print(f'Original dataset names and labels. {ds_name_list}: {self.ds_label_list}')
        else:
            self.ds_label_list = ds_labels
            print(f'Re-mapping dataset names and labels. {ds_name_list}: {self.ds_label_list}')

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.images, self.ped_labels, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ped_labels, ds_labels = [], [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)

            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ped_labels.append(contents[-1])
                ds_labels.append(ds_label)

        return images, ped_labels, ds_labels

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''
        nonPed_num, ped_num = 0, 0
        for item in self.ped_labels:
            if item == '0':
                nonPed_num += 1
            elif item == '1':
                ped_num += 1
        return nonPed_num, ped_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.ped_labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'img_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict



class noise_dataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.images = torch.randn(self.num, 3, 224, 224)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.images[idx]



# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#
#     def are_all_different(tensor_batch):
#         """
#         高效检查批次中所有张量是否不同
#         """
#         batch_size = tensor_batch.shape[0]
#         for i in range(batch_size):
#             for j in range(i + 1, batch_size):
#                 if torch.allclose(tensor_batch[i], tensor_batch[j], atol=1e-6):
#                     return False
#         return True
#
#     noise_ds = get_noise_dataset(num=1000)
#     noise_loader = DataLoader(noise_ds, batch_size=4)
#     for data in noise_loader:
#         print(data.shape)
#         flag = are_all_different(data)
#         print(flag)
#         # for image in data:
#         #     tensor_np = image.detach().numpy()
#         #     print(tensor_np.shape)
#         #
#         #     # 调整维度顺序为 [height, width, channels]
#         #     if tensor_np.shape[0] == 3:  # CHW -> HWC
#         #         tensor_np = np.transpose(tensor_np, (1, 2, 0))
#         #
#         #     # 归一化到 [0,1] 范围
#         #     tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
#         #
#         #     # plt.figure(figsize=(8, 6))
#         #     plt.imshow(tensor_np)
#         #     plt.axis('off')
#         #     plt.colorbar()
#         #     plt.show()
#
#         break




















