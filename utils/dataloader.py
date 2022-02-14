"""
    By:     hsy
    Date:   2022/1/27
"""
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2

transform = transforms.Compose([
    transforms.ToTensor()
])


"""
    Transformations for on-the-fly data augmentation 
    (Actually we didn't use on-the-fly data augmentation in training. Rather, we use the following fuctions to generate a dataset before training.)
"""
def rotate_bound(image, angle, v=0):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=v, flags=cv2.INTER_NEAREST)

def randomTransform(img, seg):
    type = random.randint(0, 2)
    borderv = [float(img[0,0,0]), float(img[0,0,1]), float(img[0,0,2]), 0]
    if type == 0:
            angle = random.randint(0, 35) * 10
            
            # plt.figure(figsize=(10,5))
            rot_img = np.zeros((240,240,3), dtype=np.float32)
            
            rot_seg = rotate_bound(seg, angle)
            bound = rot_seg.shape[0]
            
            s = random.randint(24, bound//10) * 10
            
            mid = bound // 2
            half = s // 2
            
            rot_seg = rot_seg[mid-half:mid+half, mid-half:mid+half]
            
            rot_seg = cv2.resize(rot_seg, (240, 240))
            rot_seg = (rot_seg > 0.5).astype(np.float32)
            
            for i in range(3):
                # plt.subplot(2, 4, i + 1)
                a = rotate_bound(img[:,:,i], angle, borderv[i])
                
                a = a[mid-half:mid+half, mid-half:mid+half]
                a = cv2.resize(a, (240, 240))
                rot_img[:, :, i] = a
            return rot_img, rot_seg
    else:
            scale = random.randint(24,35)*10
            
            s_seg = np.zeros((scale, scale))
            
            mid = scale // 2
            half = 120
            bias_max = mid-half
            bias1 = random.randint(-bias_max, bias_max)
            bias2 = random.randint(-bias_max, bias_max)
            s_seg[mid+bias1-half:mid+bias1+half, mid+bias2-half:mid+bias2+half] = seg
            s_seg = cv2.resize(s_seg, (240, 240))
            s_seg = (s_seg > 0.5).astype(np.float32)
            
            s_img = np.zeros((240,240,3), dtype=np.float32)
            
            for j in range(3):
                s_img_j = np.zeros((scale, scale), dtype=np.float32) + borderv[j]
                s_img_j[mid+bias1-half:mid+bias1+half, mid+bias2-half:mid+bias2+half] = img[:,:,j]
                
                s_img_j = cv2.resize(s_img_j, (240, 240))
                s_img[:,:,j] = s_img_j
            return s_img, s_seg
        

class BraTSDataset(Dataset):
    def __init__(self, data_root, test=False):
        self.data_root = data_root
        self.files = os.listdir(data_root)
        self.images = self.files[::2] if "img" in self.files[0] else self.files[1::2]
        self.test = test
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        file_path = os.path.join(self.data_root, self.images[index])
        label_path = os.path.join(self.data_root, self.images[index].replace("img", "seg"))
        img = np.load(file_path)
        label = np.load(label_path).astype(np.float32)
        # if not self.test:
        #     img, label = randomTransform(img, label)
        return transform(img), transform(label) # (3, 240, 240) tensor, (240, 240) tensor
    
    def get_img_size(self):
        H, W, C = np.load(os.path.join(self.data_root, self.images[0])).shape
        return H, W
        

if __name__ == "__main__":
    a = np.array([1,2])
    print(a.dtype)

