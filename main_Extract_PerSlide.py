import torch
import argparse
import os
from torch.utils.data import Dataset

import torchvision.transforms as T
import pickle
from Model.resnet import resnet50_baseline
from resnet import resnet18
import glob
import PIL.Image as Image
import pandas as pd

parser = argparse.ArgumentParser(description='abc')
parser.add_argument('--data_dir', default='../../dataSets/YD/APT_256/train', type=str) #### ####
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_worker', default=4, type=int)
parser.add_argument('--crop', default=224, type=int)
parser.add_argument('--batch_size_v', default=64, type=int)
parser.add_argument('--log_dir', default='../../dataSets/YD', type=str) #### ####
parser.add_argument('--img_resize', default=256, type=int)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def main():
    args = parser.parse_args()
    # feat_extractor = resnet50_baseline(pretrained=True).to(args.device)
    feat_extractor = resnet18(pretrained=True).to(args.device)

    normalize = T.Normalize(mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181], std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114])
    test_transform = T.Compose([
        T.CenterCrop(args.crop),    #RandomCrop(224),
        T.ToTensor(),
        normalize,
    ])

    all_dataset = Patch_dataset_SlideFolder_noLabel(args.data_dir, test_transform)

    all_loader = torch.utils.data.DataLoader(
        all_dataset, batch_size=args.batch_size_v, shuffle=False,
        num_workers=args.num_worker, pin_memory=True)

    extract_save_features(extractor=feat_extractor, loader=all_loader, params=args,
                          save_path=args.log_dir)


def extract_save_features(extractor, loader, params, save_path=''):

    extractor.eval()

    mDATA = {}
    count = 0

    for idx, batchdata in enumerate(loader):

        samples = batchdata['image'].to(params.device)
        slide_names = batchdata['slide_name']
        file_names = batchdata['file_name']
        #bs, chn, ww, hh = samples.shape
        BS = samples.size()[0]
        feat = extractor(samples)

        feat_np = feat.cpu().data.numpy()

        for idx, tSlideName in enumerate(slide_names):
            if tSlideName not in mDATA.keys():
                mDATA[tSlideName] = []
            tFeat = feat_np[idx]
            tFileName = file_names[idx]
            tdata = {'feature': tFeat, 'file_name': tFileName}
            mDATA[tSlideName].append(tdata)

            count += 1
            if count % 10000 == 0:
                print('curent count :',count)
    print('number count:',count)

    if save_path != '':
        # for sst in mDATA.keys():
        #     slide_save_path = os.path.join(save_path, sst+'.pkl')
        #     with open(slide_save_path, 'wb') as f:
        #         pickle.dump(mDATA[sst], f)
        slide_save_path = os.path.join(save_path, 'train_256APT'+'.pkl')
        with open(slide_save_path, 'wb') as f:
            pickle.dump(mDATA, f)

class Patch_dataset_SlideFolder_noLabel(Dataset):
    def __init__(self, slide_dir, transform=None, img_resize=256, surfix='jpg'):

        self.img_resize = img_resize

        SlideNames = os.listdir(slide_dir)
        SlideNames = [sst for sst in SlideNames if os.path.isdir( os.path.join(slide_dir, sst) ) ]

        self.patch_dirs = []
        for tslideName in SlideNames:
            tpatch_paths = glob.glob( os.path.join(slide_dir, tslideName, '*.'+surfix))
            self.patch_dirs.extend(tpatch_paths)
        self.transform = transform

    def __getitem__(self, index):

        img_dir = self.patch_dirs[index]
        timage = Image.open(img_dir).convert('RGB')
        timage = timage.resize((self.img_resize, self.img_resize))

        file_name = os.path.basename(img_dir).split('.')[0]
        tinfo = img_dir.split('/')
        slide_name = tinfo[-2]

        if self.transform is not None:
            timage = self.transform(timage)

        return {'image': timage,  'slide_name': slide_name, 'file_name': os.path.basename(img_dir) }

    def __len__(self):
        return len(self.patch_dirs)


if __name__ == '__main__':
    main()

