import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from openTSNE import TSNE
from scipy.spatial.distance import euclidean, cdist
from scipy.special import softmax, kl_div
from scipy.stats import entropy
from sklearn.metrics.pairwise import polynomial_kernel
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import convert_weights

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES, IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

class Feature_Trans_Module_two_layer(nn.Module):
    def __init__(self, input_dim=100, out_dim=256):
        super(Feature_Trans_Module_two_layer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 1)
        )

    def forward(self, input_feat):
        final_feat = self.conv1(input_feat.unsqueeze(-1).unsqueeze(-1))

        return final_feat.squeeze(-1).squeeze(-1)

class TeacherPromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.EKDA.N_CTX
        ctx_init = cfg.TRAINER.EKDA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.EKDA.TP
        self.t_deep = cfg.TRAINER.EKDA.T_DEEP
        self.v_deep = cfg.TRAINER.EKDA.V_DEEP
        self.location = cfg.TRAINER.EKDA.LOCATION
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)

        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:  # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)

        vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
        nn.init.normal_(vctx_vectors, std=0.02)
        self.vctx = nn.Parameter(vctx_vectors)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of student model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(cfg.GPU))
        clip_model_temp = load_clip_to_cpu(cfg, teacher_model=True).float().to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            all_teacher_features = []
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.to(self.device))
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        self.dim = clip_model.text_projection.shape[1]

    def forward(self):
        vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts, vctx

class StudentPromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.EKDA.N_CTX
        ctx_init = cfg.TRAINER.EKDA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.EKDA.TP
        self.t_deep = cfg.TRAINER.EKDA.T_DEEP
        self.v_deep = cfg.TRAINER.EKDA.V_DEEP
        self.location = cfg.TRAINER.EKDA.LOCATION
        self.hidden_size = clip_model.visual.conv1.weight.shape[0]  # visual encoder hiden size(768)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.num_tokens = cfg.TRAINER.EKDA.N_CTX

        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:  # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)

        vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
        nn.init.normal_(vctx_vectors, std=0.02)
        self.vctx = nn.Parameter(vctx_vectors)

        deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
        nn.init.normal_(deep_vctx_vectors, std=0.02)
        self.deep_vctx = nn.Parameter(deep_vctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of student model context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)

        self.device = torch.device("cuda:{}".format(cfg.GPU))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

        self.dim = clip_model.text_projection.shape[1]

    def forward(self):

        vctx = self.vctx

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        deep_vctx = self.deep_vctx

        return prompts, vctx, deep_vctx


class TeacherModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.image_encoder = ImageEncoder_Trans(cfg, clip_model)

        self.prompt_learner = TeacherPromptLearner(cfg, classnames, clip_model)
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.logit_scale = clip_model.logit_scale

        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)

        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.GPU))

        self.n_cls = len(classnames)
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, image, label=None):

        logit_scale = self.logit_scale.exp()

        prompts, vctx = self.prompt_learner()

        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype), vctx)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                zero_shot_logits = logit_scale * zero_shot_features.to(self.device) @ fixed_embeddings.half().to(
                    self.device).t()

            return F.cross_entropy(logits, label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits, image_features, text_features


class StudentModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model, t_dim):
        super().__init__()

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)

        self.prompt_learner = StudentPromptLearner(cfg, classnames, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)
        self.dim = clip_model.text_projection.shape[1]
        self.t_dim = t_dim

        self.VPT_image_trans = Feature_Trans_Module_two_layer(self.dim, self.t_dim)
        self.VPT_image_trans = self.VPT_image_trans
        convert_weights(self.VPT_image_trans)

        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.GPU))

    def forward(self, image, text_features):
        _, vctx, deep_vctx = self.prompt_learner()

        image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)
        image_features = self.VPT_image_trans(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.to(image_features.device).t()

        return logits, image_features

@TRAINER_REGISTRY.register()
class EKDA(BaseDA):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL
        self.temperature = cfg.TRAINER.EKDA.TEMPERATURE
        self.n_cls = len(classnames)
        self.accuracy = []

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu(cfg, True)

        if cfg.TRAINER.EKDA.PREC == "fp32" or cfg.TRAINER.EKDA.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16
            clip_model_teacher.float()

        print("Building custom CLIP...")
        self.teacher_model = TeacherModel(cfg, classnames, clip_model_teacher)
        self.student_model = StudentModel(cfg, classnames, clip_model, self.teacher_model.dim)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
        for name, param in self.student_model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "VPT" in name:
                param.requires_grad_(True)

        Teacher_Total_Memory = 0
        for name, param in self.teacher_model.named_parameters():
            if param.requires_grad:
                Teacher_Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Teacher Model Total Memory : " + str(Teacher_Total_Memory) + "MB")

        Student_Total_Memory = 0
        for name, param in self.student_model.named_parameters():
            if param.requires_grad:
                Student_Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Student Model Total Memory : " + str(Student_Total_Memory) + "MB")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.student_model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        self.trainable_list_t = nn.ModuleList([])
        self.trainable_list_s = nn.ModuleList([])
        self.trainable_list_t.append(self.teacher_model.prompt_learner)
        self.trainable_list_s.append(self.student_model.prompt_learner)
        self.trainable_list_s.append(self.student_model.VPT_image_trans)

        # NOTE: only give prompt_learner to the optimizer
        self.optimizer_T = build_optimizer(self.trainable_list_t, cfg.OPTIM)
        self.optimizer_S = build_optimizer(self.trainable_list_s, cfg.OPTIM)

        self.sched_T = build_lr_scheduler(self.optimizer_T, cfg.OPTIM)
        self.sched_S = build_lr_scheduler(self.optimizer_S, cfg.OPTIM)

        self.register_model("TeacherPromptLearner", self.trainable_list_t, self.optimizer_T, self.sched_T)
        self.register_model("StudentPromptLearner", self.trainable_list_s, self.optimizer_S, self.sched_S)


    def forward_backward(self, batch_x, batch_u):

        image_x, label_x, image_u, label_u = self.parse_batch_train(batch_x, batch_u)

        self.teacher_model.train()
        self.student_model.train()

        """Train Teacher Model
        """
        self.optimizer_T.zero_grad()
        loss_ce, text_features, zs_text_features, zs_image_features, image_features, \
        zs_logits, logits = self.teacher_model(image_x, label_x)

        loss_scl_text = F.l1_loss(text_features, zs_text_features.to(self.device),
                                  reduction='mean') * self.cfg.TRAINER.EKDA.TEXT_LOSS_WEIGHT
        # Calculate the L_SCL_image loss
        loss_scl_image = F.l1_loss(image_features, zs_image_features.to(self.device),
                                   reduction='mean') * self.cfg.TRAINER.EKDA.IMAGE_LOSS_WEIGHT
        # Now calculate L_SCL_logits
        l_scl_logits = F.kl_div(
            F.log_softmax(logits, dim=1),
            F.log_softmax(zs_logits, dim=1),
            reduction='sum',
            log_target=True
        ) / logits.numel()

        loss_scl = (l_scl_logits + loss_scl_text + loss_scl_image)
        loss_teacher = loss_ce + loss_scl
        loss_teacher.backward()

        self.optimizer_T.step()

        """Train Student Model
        """

        self.teacher_model.eval()
        self.optimizer_S.zero_grad()

        logits_teacher, image_features_t, text_features = self.teacher_model(image_u)
        logits_student, image_features_s = self.student_model(image_u, text_features)

        loss_kl = F.kl_div(
            F.log_softmax(logits_student / self.temperature, dim=1),
            F.softmax(logits_teacher / self.temperature, dim=1),
            reduction='sum',
        ) * (self.temperature * self.temperature) / logits_student.numel()

        pseudo_label = torch.softmax(logits_teacher, dim=-1)
        max_probs, label_p = torch.max(pseudo_label, dim=-1)
        loss_ce = F.cross_entropy(logits_student, label_p)

        loss_align = align_loss(image_features_t, image_features_s)

        loss_student = self.cfg.TRAINER.EKDA.KD_WEIGHT * loss_kl + loss_ce + loss_align
        loss_student.backward()

        self.optimizer_S.step()

        loss_summary = {
            "loss_s": loss_teacher.item(),
            "loss_t": loss_student.item(),
            "acc_s": compute_accuracy(logits_teacher, label_u)[0].item(),
            "acc_t": compute_accuracy(logits_student, label_u)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        label_u = batch_u["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        label_u = label_u.to(self.device)
        return input, label, input_u, label_u

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        # self.set_model_mode("eval")
        self.teacher_model.eval()
        self.student_model.eval()

        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        # only one
        self.text_features = None
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if batch_idx == 0:
                _,_, text_features = self.teacher_model(input)
                self.text_features = text_features
            output, _ = self.student_model(input, self.text_features)

            self.evaluator.process(output, label)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        self.accuracy.append(round(results["accuracy"], 1))
        print(self.accuracy)

        return list(results.values())[0]


