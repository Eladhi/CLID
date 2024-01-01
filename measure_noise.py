import argparse
import json
import logging
import os
import re
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.modeling_bert import BertConfig

from config_denoise import _C as config
from dataset_denoise import COCOCaptionDataset
from modeling import Generator
from utils import mkdir
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from utils.tokenizer import EOS, MASK, tokenizer, PAD


def calc_data_prob(generator, data_loader, device, stage, past_only):
    generator.eval()

    pred_dict = dict()
    eos_penalizers = list()
    for l, (low, high) in enumerate(config.boundaries):
        pred_dict[str(l + 1)] = dict()

        eos_penalizer = torch.ones((1, high - low + 1), dtype=torch.float, device=device)
        eos_penalizers.append(eos_penalizer)

    for iteration, batch in tqdm(enumerate(data_loader, 0), total=len(data_loader)):

        token_type_ids = batch[0].to(device)  # (N, L), long
        input_token_ids = batch[1].to(device)  # (N, L), long
        masked_token_ids = batch[2].to(device)  # (N, L), long
        region_features = batch[3].to(device)  # (N, 100, 2048), float
        region_class = batch[4].to(device)  # (N, 100, 1601), float
        region_spatial = batch[5].to(device)  # (N, 100, 6), float
        img_idx = batch[6].to(device)

        num_img_tokens = region_spatial.size(1)
        seq_length = input_token_ids.size(1)
        batch_size = input_token_ids.size(0)
        B = region_class.size(0)
        num_regions = region_class.size(1)

        with torch.no_grad():
            batch_id = torch.arange(0, B, 1, device=device).unsqueeze(1)
            region_spatial[:, :, [0, 2]] /= region_spatial[:, :, [2]] + 1e-5
            region_spatial[:, :, [1, 3]] /= region_spatial[:, :, [3]] + 1e-5
            rel_area = (region_spatial[:, :, [3]] - region_spatial[:, :, [1]]) * \
                       (region_spatial[:, :, [2]] - region_spatial[:, :, [0]])
            region_spatial = torch.cat((region_spatial[:, :, :4],
                rel_area.clamp_(0), region_spatial[:, :, 5:]), dim=-1)
            position_features = torch.cat((F.layer_norm(region_spatial, [6]),
                F.layer_norm(region_class, [1601])), dim=-1)

            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_token_ids)

            region_type = position_ids.new_full(
                region_features.shape[:2], len(config.boundaries) + 1)
            token_type_ids = torch.cat((region_type, token_type_ids), dim=1)

            attention_mask = (masked_token_ids != PAD).float()
            _attention_mask = attention_mask.new_ones((batch_size, num_img_tokens))
            attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

            masked_token_ids[masked_token_ids > 0] = MASK
            pred_probs_cond = torch.zeros_like(masked_token_ids).to(torch.float32)

            if past_only:
                for i in range((input_token_ids != PAD).sum(dim=1).max()):
                    masked_token_ids[:, :i] = input_token_ids[:, :i]

                    pred_scores = generator(
                        region_features, position_features,
                        masked_token_ids, token_type_ids,
                        position_ids, attention_mask)

                    pred_probs = F.softmax(pred_scores[:, num_regions:, :], dim=-1)
                    pred_probs = torch.gather(pred_probs, dim=2, index=input_token_ids.unsqueeze(2)).squeeze()
                    pred_probs_cond[:, i] = pred_probs[:, i]

            else:
                for i in range((input_token_ids != PAD).sum(dim=1).max()):
                    masked_token_ids = input_token_ids.detach().clone()
                    masked_token_ids[:, i] = MASK

                    pred_scores = generator(
                        region_features, position_features,
                        masked_token_ids, token_type_ids,
                        position_ids, attention_mask)

                    pred_probs = F.softmax(pred_scores[:, num_regions:, :], dim=-1)
                    pred_probs = torch.gather(pred_probs, dim=2, index=input_token_ids.unsqueeze(2)).squeeze()
                    pred_probs_cond[:, i] = pred_probs[:, i]

            pred_log_probs = torch.log(pred_probs_cond)
            pred_log_probs *= (input_token_ids > 0)
            pred_log_probs = torch.sum(pred_log_probs, dim=1) / (input_token_ids > 0).sum(dim=1)  # normalization

            # place results in the dataset
            for i in range(pred_probs.shape[0]):
                data_loader.dataset.update_sample_scores(int(img_idx[i]), float(pred_log_probs[i]), diff=stage)


def save_scores(data_loader):
    data_loader.dataset.save_scores()


def load_scores(data_loader):
    data_loader.dataset.load_scores()


def evaluate_with_model(data_loader, model_path, stage):

    device = torch.device(config.device)
    num_types = len(config.boundaries) + 2

    generator = Generator(BertConfig(type_vocab_size=num_types))
    generator = generator.to(device)
    g_checkpointer = Checkpointer(model=generator)
    g_checkpointer.load(model_path, True)

    calc_data_prob(generator, data_loader, device, stage, past_only=True)
