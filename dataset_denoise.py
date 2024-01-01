import json
import os.path as osp
import random
import numpy as np
import time
from collections import namedtuple

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.tokenizer import EOS, MASK, PAD, tokenizer

Sample = namedtuple("Sample", ["caption", "image_id", "real"])

class COCOCaptionDataset(Dataset):

    def __init__(self, root, split, boundaries, arguments):
        self.split = split
        self.root = root
        self.boundaries = boundaries
        self.len_cnt = np.zeros(len(self.boundaries))
        self.length_sorted = None
        self.length_sorted_trusted = None
        self.arguments = arguments

        if self.split == 'test':
            self.load_fn = self._get_item_infer
            self.build_infer_samples()
        else:
            self.load_fn = self._get_item_train
            self.training_phase = 0
            self.build_train_samples()

    def build_infer_samples(self):
        self.samples = {}
        with open(osp.join(self.root, 'dataset_coco_trusted.json')) as f:
            captions = json.load(f)
            captions = captions['images']

        file_id2captions_test = osp.join(self.root, 'id2captions_test.json')
        file_test_samples = osp.join(self.root, 'test_samples.json')
        if not osp.exists(file_id2captions_test):
            samples = list()
            id2captions = dict()
            for item in captions:
                if item['split'] in self.split:
                    image_id = item['filename'].split('.')[0]
                    samples.append(image_id)
                    image_id = str(int(image_id[-12:]))
                    id2captions[image_id] = list()
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        id2captions[image_id].append({'caption': caption})

            with open(file_id2captions_test, 'w') as f:
                json.dump(id2captions, f)
            with open(file_test_samples, 'w') as f:
                json.dump({'ids': samples}, f)
        else:
            with open(file_test_samples) as f:
                samples = json.load(f)['ids']

        self.samples['chosen'] = samples

    def build_train_samples(self):
        # create both trusted and noisy data files
        self.samples = {}
        for dataset_type in ['trusted', 'noisy']:
            with open(osp.join(self.root, 'dataset_coco_' + dataset_type + '.json')) as f:
                captions = json.load(f)
                captions = captions['images']
            file_train_data = osp.join(self.root, f'{self.split}_data_' + dataset_type + '.pth')
            if not osp.exists(file_train_data):
                print('Creating .pth ' + dataset_type + ' data file')
                samples = list()
                for item in captions:
                    if item['split'] in self.split:
                        image_id = item['filename'].split('.')[0]
                        for ci, c in enumerate(item['sentences']):
                            caption = ' '.join(c['tokens']) + '.'
                            caption = tokenizer.encode(caption)
                            if len(caption) > self.boundaries[-1][-1]:
                                continue
                            real = 1 if ci < 5 else 0
                            sample = Sample(caption=caption, image_id=image_id, real=real)
                            samples.append(sample)
                torch.save(samples, file_train_data)
            else:
                samples = torch.load(file_train_data)
            self.samples[dataset_type] = samples

        self.samples['chosen'] = samples
        self.length_sorted = np.array([len(s[0]) for s in self.samples['chosen']]).argsort()[::-1]
        self.length_sorted_trusted = np.array([len(s[0]) for s in self.samples['trusted']]).argsort()[::-1]

        file_train_scores = osp.join(self.root, f'{self.split}_data_scored.pth')
        if not osp.exists(file_train_scores):
            self.scored = {}
        else:
            self.scored = torch.load(file_train_scores)

    def set_train_phase(self, phase, logger):
        self.training_phase = phase
        if self.training_phase == 0:
            logger.info('Setting noisy dataset for training')
            self.samples['chosen'] = self.samples['noisy']
        elif self.training_phase == 1:
            logger.info('Setting trusted dataset for training')
            self.samples['chosen'] = self.samples['trusted']
        else:
            logger.info('Perform sampling from the noisy dataset')
            self.score_based_sample(logger)
            logger.info('Sampled %d images' % len(self.samples['chosen']))
            logger.info('Mean Score: %.2f' % np.array(self.samples['chosen_scores']).mean())
        self.length_sorted = np.array([len(s[0]) for s in self.samples['chosen']]).argsort()[::-1] # argsort, no shuffle, then map index in the loading & update

    def update_sample_scores(self, index, p, diff=False):
        if not diff:
            self.scored[index] = [self.samples['chosen'][index][0], self.samples['chosen'][index][1], p, self.samples['chosen'][index][2]]
        else:
            prev_p = self.scored[index][2]
            self.scored[index] = [self.samples['chosen'][index][0], self.samples['chosen'][index][1], prev_p-p, self.samples['chosen'][index][2]]

    def save_scores(self):
        file_train_scores = osp.join(self.root, f'{self.split}_data_scored.pth')
        torch.save(self.scored, file_train_scores)

    def load_scores(self):
        file_train_scores = osp.join(self.root, f'{self.split}_data_scored.pth')
        self.scored = torch.load(file_train_scores)

    def score_based_sample(self, logger):
        # In this implementations, we measure noise rather than quality (in oppose to the paper).
        # High quality means low noise.

        full_idx_list = np.array(list(self.scored.keys()))
        rel_idx_list = np.arange(full_idx_list.shape[0])
        score_list = np.array([v[2] for v in self.scored.values()])

        tau = self.arguments['config']['scheduler']['s']
        reduce_p = self.arguments['config']['scheduler']['p']

        trusted = np.array([v[3] for v in self.scored.values()])
        Bt_rel = rel_idx_list
        Bt_idx = full_idx_list[Bt_rel]
        Bt_scores = score_list[Bt_rel]
        synth_idx_rel = Bt_rel[trusted == 0]
        synth_idx = full_idx_list[synth_idx_rel]
        synth_scores = score_list[synth_idx_rel]
        percent_left = np.max((0, np.round(100 - (reduce_p * (self.training_phase - 2)))))
        thresh = np.percentile(synth_scores, percent_left)  # reduce p% every step
        print('Sampling Threshold Synth Only: %.2f' % thresh)
        synth_probs = 1 - 0.5 * (1 + np.tanh((synth_scores - thresh) / tau))
        synth_probs /= synth_probs.sum()
        if percent_left > 0:
            sample_idx = np.random.choice(synth_idx, int(synth_idx.shape[0] * (1 - (reduce_p * (self.training_phase - 2)) / 100)), replace=False, p=synth_probs)
        else:
            sample_idx = np.random.choice(synth_idx, 0)
        trusted_idx = Bt_idx[trusted == 1]
        sample_idx = np.unique(np.concatenate((sample_idx, trusted_idx)))

        logger.info('Step variance: %.2f' % tau)
        logger.info('Step threshold: %.2f' % thresh)
        logger.info('Num captions: %d' % sample_idx.shape[0])

        self.samples['chosen'] = [self.samples['noisy'][i] for i in sample_idx]
        self.samples['chosen_scores'] = [self.scored[s][2] for s in sample_idx]

        self.len_cnt *= 0
        for s in self.samples['chosen']:
            for i, l in enumerate(self.boundaries):
                if l[0] <= len(s[0]) <= l[1]:
                    self.len_cnt[i] += 1
                    break
        logger.info('Num captions per level: %s' % np.array2string(self.len_cnt, separator=', '))
        logger.info('Percentage captions per level: %s' % np.array2string(self.len_cnt / self.len_cnt.sum(), precision=5, separator=', '))

    def get_region_feature(self, name):
        with h5py.File(osp.join(self.root, '..', 'region_feat_gvd_wo_bgd', 'feat_cls_1000', f'coco_detection_vg_100dets_gvd_checkpoint_trainval_feat{name[-3:]}.h5'), 'r') as features, \
                h5py.File(osp.join(self.root, '..', 'region_feat_gvd_wo_bgd', 'feat_cls_1000', f'coco_detection_vg_100dets_gvd_checkpoint_trainval_cls{name[-3:]}.h5'), 'r') as classes, \
                h5py.File(osp.join(self.root, '..', 'region_feat_gvd_wo_bgd', f'coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5'), 'r') as bboxes:
            region_feature = torch.from_numpy(features[name][:])
            region_class = torch.from_numpy(classes[name][:])
            region_spatial = torch.from_numpy(bboxes[name][:])
        return region_feature, region_class, region_spatial

    def __getitem__(self, index):
        return self.load_fn(index)

    def _get_item_train(self, index):

        # use random index as captions are sorted by length
        index = self.length_sorted[index]
        sample = self.samples['chosen'][index]
        input_token_id = sample.caption

        length = len(input_token_id)
        for i, (l, h) in enumerate(self.boundaries, 1):
            if l <= length <= h:
                length_level = i
                break

        high = self.boundaries[length_level - 1][1]
        offset = high - length

        token_type_id = [length_level] * high
        token_type_id = torch.tensor(token_type_id, dtype=torch.long)

        input_token_id = input_token_id + [EOS] * offset

        masked_token_id = input_token_id.copy()
        num_masks = random.randint(max(1, int(0.1 * high)), high)
        selected_idx = random.sample(range(high), num_masks)
        for i in selected_idx:
            masked_token_id[i] = MASK

        input_token_id = torch.tensor(input_token_id, dtype=torch.long)
        masked_token_id = torch.tensor(masked_token_id, dtype=torch.long)

        region_feature, region_class, region_spatial = \
            self.get_region_feature(sample.image_id)

        return token_type_id, input_token_id, masked_token_id, \
               region_feature, region_class, region_spatial, index, sample.real

    def _get_item_infer(self, index):
        sample = self.samples['chosen'][index]
        region_feature, region_class, region_spatial = \
            self.get_region_feature(sample)
        image_id = torch.tensor(int(sample[-12:]), dtype=torch.long)
        return region_feature, region_class, region_spatial, image_id

    def __len__(self):
        return len(self.samples['chosen'])


def collate_fn_train(batch):
    batch = list(zip(*batch))

    token_type_id = pad_sequence(batch[0], batch_first=True, padding_value=0)
    input_token_id = pad_sequence(batch[1], batch_first=True, padding_value=PAD)
    masked_token_id = pad_sequence(batch[2], batch_first=True, padding_value=PAD)

    region_feature = torch.stack(batch[3], dim=0)
    region_class = torch.stack(batch[4], dim=0)
    region_spatial = torch.stack(batch[5], dim=0)

    index = torch.tensor(batch[6])
    real = torch.tensor(batch[7])

    return token_type_id, input_token_id, masked_token_id, \
           region_feature, region_class, region_spatial, index, real
