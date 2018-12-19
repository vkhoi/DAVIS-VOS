import cv2
import davis
import h5py
import numpy as np
import os

from .helper import read_flow

from torch.utils.data import Dataset
from torchvision import transforms

class Davis2017(Dataset):
    def __init__(self, train=True,
                 directory="/vulcan/scratch/khoi/databases/davis-2017/data/DAVIS",
                 # directory="/Users/khoipham/Documents/umd/research/workspace/databases/davis-2017/data/DAVIS",
                 transform=None,
                 fname=None,
                 mode="app"):
        self.train = train
        self.directory = directory
        self.transform = transform
        self.mode = mode

        self.first_frame = set()
        self.last_frame = set()

        if train:
            file_split_name = "train"
        else:
            file_split_name = "val"

        images = []
        gts = []
        # 2 types of optical flow: raw & intensity.
        if "flow-raw" in mode:
            flows_raw = []
        if "flow-intensity" in mode:
            flows = []

        with open(os.path.join(directory, "ImageSets", "2017", 
                               file_split_name + ".txt")) as f:
            lines = f.readlines()

            for img_name in lines:
                img_name = img_name.strip()
                # print(img_name)

                if fname is not None and fname != img_name:
                    continue

                img_folder_name = os.path.join("JPEGImages", "480p", img_name)
                gt_folder_name = os.path.join("Annotations", "480p", img_name)

                img_list = sorted(os.listdir(os.path.join(directory, 
                                                          img_folder_name)))
                self.first_frame.add(len(images))
                images += list(map(
                    lambda x: os.path.join(img_folder_name, x), img_list))
                self.last_frame.add(len(images) - 1)

                img_list = sorted(os.listdir(os.path.join(directory, 
                                                          gt_folder_name)))
                gts += list(map(
                    lambda x: os.path.join(gt_folder_name, x), img_list))

                if "flow-raw" in mode:
                    flow_raw_folder = os.path.join("Flows", "480p", img_name)
                    flow_raw_list = sorted(
                        os.listdir(os.path.join(directory, flow_raw_folder)))
                    flow_raw_list = list(map(
                        lambda x: os.path.join(flow_raw_folder, x), flow_raw_list))
                    
                    # There are only N-1 flows computed from video of N frames.
                    # Add one dummy flow to make the list have length N.
                    flow_raw_list.append(flow_raw_list[len(flow_raw_list) - 1])
                    flows_raw += flow_raw_list

                if "flow-intensity" in mode:
                    flow_intensity_folder = os.path.join("Flows_Intensity", 
                                                         "480p", img_name)
                    flow_intensity_list = sorted(
                        os.listdir(os.path.join(directory, flow_intensity_folder)))
                    flow_intensity_list = list(map(
                        lambda x: os.path.join(flow_intensity_folder, x), 
                                               flow_intensity_list))
                    flows += flow_intensity_list

            assert len(images) == len(gts)
        self.images = images
        self.gts = gts
        if "flow-raw" in mode:
            self.flows_raw = flows_raw
        if "flow-intensity" in mode:
            self.flows = flows

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx, keep_gt=False, instance_id=None):
        image = cv2.imread(os.path.join(self.directory, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Sanity check. Resize to (224, 224) and forward through VGG16
        # to see if VGG classifies correctly.
        # image = cv2.resize(image, (224, 224))

        image = np.array(image, dtype=np.float32)
        image /= 255.0
        # image /= np.max([image.max(), 1e-8])
        # image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.astype(np.float32)

        # gt = cv2.imread(os.path.join(self.directory, self.gts[idx]), 0)
        gt = davis.io.imread_indexed(os.path.join(self.directory,
                                                  self.gts[idx]))[0]
        # num_objs = np.max(gt)
        # rand_obj = np.random.randint(low=1, high=num_objs + 1)
        # gt[gt != rand_obj] = 0
        if not keep_gt:
            if instance_id is None:
                gt = np.array(gt, dtype=np.float32)
                gt[gt > 0] = 1
            else:
                gt = np.array(gt, dtype=np.uint8)
                gt[gt != instance_id] = 0
                gt[gt == instance_id] = 1
                gt = np.array(gt, dtype=np.float32)

        # gt = np.array(gt, dtype=np.float32)
        # gt[gt > 0] = 1
        # gt /= np.max([gt.max(), 1e-8])

        sample = {
            "image": image,
            "gt": gt
        }

        # if not self.is_first_frame(idx):
        #     gt_prev = davis.io.imread_indexed(os.path.join(
        #         self.directory,self.gts[idx - 1]))[0]
        #     gt_prev = np.array(gt_prev, dtype=np.float32)
        #     gt_prev[gt_prev > 0] = 1
        #     sample["gt_prev"] = gt_prev
        # else:
        #     sample["gt_prev"] = gt.copy()

        if "flow-raw" in self.mode:
            if idx == 0:
                flow_raw = np.zeros((480, 854), dtype=np.float32)
            else:
                # print(self.flows_raw)
                flow_raw = read_flow(os.path.join(self.directory, self.flows_raw[idx-1]))
                flow_raw[:,:,0] /= flow_raw.shape[1]
                flow_raw[:,:,1] /= flow_raw.shape[0]
            sample["flow_raw"] = flow_raw
        
        if "flow-intensity" in self.mode:
            with h5py.File(os.path.join(self.directory, self.flows[idx]), "r") as hf:
                flow = hf["data"][:]
                sample["flow_intensity"] = flow

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getimg__(self, idx):
        image = cv2.imread(os.path.join(self.directory, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def is_last_frame(self, idx):
        if idx in self.last_frame:
            return True
        return False

    def is_first_frame(self, idx):
        if idx in self.first_frame:
            return True
        return False

    def make_flow_intensity_dataset(self):
        for i in range(len(self.images)):
            print(self.images[i])
            tokens = self.images[i].split("/")
            fname = tokens[2]
            order = tokens[3].split(".")[0]

            flow_path = self.flows_raw[i]
            if i > 0:
                flow_rev_path = self.flows_raw[i - 1].replace("Flows", "Flows_Reverse")
            else:
                flow_rev_path = None

            flow_forward = read_flow(os.path.join(self.directory, flow_path))
            if flow_rev_path is not None:
                flow_backward = read_flow(os.path.join(self.directory, flow_rev_path))

            flow = np.zeros((flow_forward.shape[0], flow_forward.shape[1], 3))

            if self.is_first_frame(i):
                flow[:,:,0] = np.sqrt(flow_forward[:,:,0]**2 + flow_forward[:,:,1]**2)
                flow[:,:,1] = flow[:,:,0]
                flow[:,:,2] = flow[:,:,0]
            elif self.is_last_frame(i):
                flow[:,:,1] = np.sqrt(flow_backward[:,:,0]**2 + flow_backward[:,:,1]**2)
                flow[:,:,0] = flow[:,:,1]
                flow[:,:,2] = flow[:,:,1]
            else:
                flow[:,:,0] = np.sqrt(flow_forward[:,:,0]**2 + flow_forward[:,:,1]**2)
                flow[:,:,1] = np.sqrt(flow_backward[:,:,0]**2 + flow_backward[:,:,1]**2)
                flow[:,:,2] = (flow[:,:,0] + flow[:,:,1]) / 2
            
            flow = flow.astype(np.float32)

            if not os.path.exists(os.path.join(self.directory, "Flows_Intensity",
                                               "480p", fname)):
                os.makedirs(os.path.join(self.directory, "Flows_Intensity",
                                         "480p", fname))
            save_path = os.path.join(self.directory, "Flows_Intensity", "480p",
                                     fname, order + ".h5")
            h5_flow = h5py.File(save_path, "w")
            h5_flow.create_dataset("data", data=flow)
            h5_flow.close()
