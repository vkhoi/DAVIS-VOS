import cv2
import numpy as np
import random
import torch

class ToTensor(object):
    def __call__(self, sample):
        for key, val in sample.items():
            if key == "gt":
                val = val[:, :, np.newaxis]

            if key == "gt" or key == "image" or key == "flow_intensity":
                val = val.transpose([2, 0, 1])

            sample[key] = torch.from_numpy(val)

        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly 
    with a probability of 0.5."""

    def __call__(self, sample):
        if random.random() < 0.5:
            for key, val in sample.items():
                img = cv2.flip(sample[key], flipCode=1)
                
                if key == "flow_raw":
                    img = -img

                sample[key] = img

        return sample

class RandomColorIntensity(object):
    def __call__(self, sample):
        img = sample["image"]

        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        v = np.random.uniform(low=0.75, high=1.5)
        img[:,:,2] = img[:,:,2] * v
        img[:,:,2][img[:,:,2] > 255]  = 255

        v = np.random.uniform(low=0.75, high=1.5)
        img[:,:,1] = img[:,:,1] * v
        img[:,:,1][img[:,:,1] > 255]  = 255

        v = np.random.uniform(low=0.9, high=1.1)
        img[:,:,0] = img[:,:,0] * v
        img[:,:,0][img[:,:,0] > 179]  = 179

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img / 255).astype(np.float32)

        sample["image"] = img

        return sample