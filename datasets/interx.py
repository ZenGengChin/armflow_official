import codecs as cs
import h5py
import numpy as np
import random
import os
import torch
from os.path import join as pjoin
from torch.utils import data
from tqdm import tqdm
import utils.rotation_conversions as geometry
from utils.body_model import BodyModel
from utils.interx_utils import InterxNormalizerTorch
from torch.utils.data._utils.collate import default_collate
from utils.word_vectorizer import WordVectorizer

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def interx_collate(batch):
    word_embeddings = torch.stack([torch.from_numpy(item[0]) for item in batch])
    pos_one_hots = torch.stack([torch.from_numpy(item[1]) for item in batch])
    captions = [item[2] for item in batch]
    sent_lens = torch.tensor([torch.tensor(item[3]) for item in batch], dtype=torch.long)
    motions = torch.stack([(item[4]) for item in batch])
    lengths = torch.tensor([torch.tensor(item[5]) for item in batch], dtype=torch.long)
    
    return {
        'text': captions,
        'cap_lens': sent_lens,
        'motions': motions,
        'motion1': motions[:, :, :, :6],
        'motion2': motions[:, :, :, 6:],
        'motion_lens': lengths,
        'word_embeddings': word_embeddings,
        'pos_one_hots': pos_one_hots,
    }


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


    
class MotionDatasetV2HHI(data.Dataset):
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.motion_rep = opt.motion_rep
        
        self.motion_file = os.path.join(opt.motion_dir, f'{split}.h5')
        
        if opt.motion_rep == "global":
            motion_file = motion_file.replace('.h5', '_global.h5')
            self.normalize = True
            self.normalizer = InterxNormalizerTorch()

        self.data = []
        self.lengths = []
        self.num_person = 1
        id_list = []
        
        split_file = os.path.join(opt.data_root, 'splits', opt.split_file)
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        with h5py.File(motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            T,J,D = mf[id_list[0]][:].astype('float32').shape
            d =  D//2
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if motion.shape[0] < opt.window_size:
                        continue
                    
                    motion1, motion2 = motion[:, :, :d], motion[:, :, d:]
                    # self.lengths.append(motion1.shape[0] - opt.window_size)
                    self.data.append(motion1)
                    # self.lengths.append(motion2.shape[0] - opt.window_size)
                    self.data.append(motion2)
                except Exception as e: 
                    print(e)
                    pass

        # self.cumsum = np.cumsum([0] + self.lengths)
        self.indices = self._create_indices()
        print("Total number of motions {}, snippets {}".format(len(self.data), len(self.indices)))

    def _create_indices(self):
        # Create a list of tuples (sample_index, time_index) for data retrieval
        indices = []
        for i, sample in enumerate(self.data):
            for start_idx in range(0, len(sample) - self.opt.window_size + 1, self.opt.window_stride):
                indices.append((i, start_idx))
        return indices

    def inv_transform(self, data):
        return data

    def __len__(self):
        # return self.cumsum[-1]
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve a window of data based on the index
        sample_idx, time_idx = self.indices[idx]
        sample = self.data[sample_idx]
        motion = sample[time_idx:time_idx + self.opt.window_size]
        
        if self.motion_rep == "smpl":
            rot_6d = self.to_rot_6d(motion[:,:-1,:]).float()

            transl = to_torch(motion[:,-1,:])
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel = torch.cat([transl, vel], axis=-1)

            motion = torch.cat([rot_6d, transl_vel.unsqueeze(1)], axis=1)
        
        elif self.motion_rep == "global":
            rot_6d = self.to_rot_6d(motion[:,:,-3:]).float()
            transl_vel = to_torch(motion[:,:,:-3])

            motion = torch.cat([transl_vel, rot_6d], axis=-1)
            
            if self.normalize:
                motion = self.normalizer.forward(motion)

        
        
        ## old 
        # if item != 0:
        #     motion_id = np.searchsorted(self.cumsum, item) - 1
        #     idx = item - self.cumsum[motion_id] - 1
        # else:
        #     motion_id = 0
        #     idx = 0
        # motion = self.data[motion_id][idx:idx+self.opt.window_size]
        # motion = self.to_rot_6d(motion).float()
        return motion #B,T,J,D

    def to_rot_6d(self, pose):
        pose = to_torch(pose)
        
        pose_all = []
        pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,0:3])))
        
        ret = torch.cat(pose_all, dim=2)
        return ret

class Text2MotionDatasetV2HHI(data.Dataset):
    def __init__(self, opt, split, w_vectorizer):
        self.opt = opt
        self.motion_rep = opt.motion_rep
        self.split_file = os.path.join(opt.data_root, 'splits', split+'.txt')
        self.motion_file = os.path.join(opt.motion_dir, f'{split}.h5')
        if opt.motion_rep == "global":
            motion_file = motion_file.replace('.h5', '_global.h5')
            self.normalize = True
            self.normalizer = InterxNormalizerTorch()

        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.num_person = 2
        min_motion_len = 24 #30 if self.opt.dataset_name =='hhi_text' else 24

        data_dict = {}
        id_list = []
        with cs.open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, 'r') as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype('float32')
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e: 
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        """ if (motion_rep == 'smpl') [T, 55, 6 + 6] (cat) 
            [T, 1, 3 trans+ 3 transvel + 3 trans + 3 transvel]
        """
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        
        ## Prepare motion
        if self.motion_rep == "smpl":
            rot_6d = self.to_rot_6d(motion[:,:-1,:]).float()

            transl = to_torch(motion[:,-1,:])
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel_all = []
            for ii in range(self.num_person):
                transl_vel_all.append(torch.cat([transl[:,3*ii:3*ii+3], vel[:,3*ii:3*ii+3]], axis=1))
            transl_vel_all = torch.cat(transl_vel_all, axis=1)

            motion = torch.cat([rot_6d, transl_vel_all.unsqueeze(1)], axis=1)
        
        elif self.motion_rep == "global":
            motion1, motion2 = motion[..., :motion.shape[-1]//2], motion[..., motion.shape[-1]//2:]

            rot_6d1 = self.to_rot_6d(motion1[:,:,-3:]).float()
            transl_vel1 = to_torch(motion1[:,:,:-3])
            motion1 = torch.cat([transl_vel1, rot_6d1], axis=-1)

            rot_6d2 = self.to_rot_6d(motion2[:,:,-3:]).float()
            transl_vel2 = to_torch(motion2[:,:,:-3])
            motion2 = torch.cat([transl_vel2, rot_6d2], axis=-1)
            
            if self.normalize:
                motion1 = self.normalizer.forward(motion1)
                motion2 = self.normalizer.forward(motion2)

            motion = torch.cat([motion1, motion2], axis=-1)
            
        elif self.motion_rep == "aa":
            '''
            rot_6d = self.to_rot_6d(motion[:,:-1,:]).float()

            transl = to_torch(motion[:,-1,:])
            vel = transl[1:] - transl[:-1]
            vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)
            transl_vel_all = []
            for ii in range(self.num_person):
                transl_vel_all.append(torch.cat([transl[:,3*ii:3*ii+3], vel[:,3*ii:3*ii+3]], axis=1))
            transl_vel_all = torch.cat(transl_vel_all, axis=1)

            motion = torch.cat([rot_6d, transl_vel_all.unsqueeze(1)], axis=1)
            
            rot_6d1 = geometry.rotation_6d_to_axis_angle(
                rot_6d[:,:,:6]
            )
            rot_6d2 = geometry.rotation_6d_to_axis_angle(
                rot_6d[:,:,6:]
            )
            motion = torch.cat([torch.cat([rot_6d1, rot_6d2], dim=-1), transl.unsqueeze(1)], dim=1)
            '''
            motion = torch.tensor(motion)
        
        if m_length < self.max_motion_length:
            motion = torch.cat([motion,
                                torch.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                ], dim=0)
        else:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

    
    def to_rot_6d(self, pose):
        # pose: T, numjoint (55), 6
        pose = to_torch(pose)

        pose_all = []
        pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,0:3])))
        if self.motion_rep == "smpl" or self.motion_rep == "aa":
            pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:,:,3:6])))
        
        ret = torch.cat(pose_all, dim=2) 
        return ret


    

class InterX(Text2MotionDatasetV2HHI):
    def __init__(self, opt, split): 
        word_vectorizer = WordVectorizer(os.path.join(opt.data_root, 'processed/glove'), 'hhi_vab')
        super().__init__(opt, split, word_vectorizer)
        
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return super().__len__()