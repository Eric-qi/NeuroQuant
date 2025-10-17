from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop

import os

# Video dataset
class VideoDataSet(Dataset):
    def __init__(self, cfg, args):
        self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]

        # Resize the input video and center crop
        self.crop_h, self.crop_w = cfg['crop_h'], cfg['crop_w']
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        
        self.diff = cfg['diff_enc']

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def img_transform(self, img):
        img = center_crop(img, (self.crop_h, self.crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        if self.diff:
            id_p = idx-1 if idx!=0 else idx
            id_f = idx+1 if idx!=len(self.video)-1 else idx
            
            tensor_image = self.img_transform(self.img_load(idx))
            tensor_image_p = self.img_transform(self.img_load(id_p))
            tensor_image_f = self.img_transform(self.img_load(id_f))
            
            sample = {
			"img_id": idx,
			"img_gt": tensor_image, #3 h w
			"img_p" : tensor_image_p, #3 h w
			"img_f" : tensor_image_f, #3 h w 
            }
            
        else:
            tensor_image = self.img_transform(self.img_load(idx))
            norm_idx = float(idx) / len(self.video)
            sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        
        return sample
    
    
