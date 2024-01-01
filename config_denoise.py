from yacs.config import CfgNode as CN

_C = CN()
_C.device = 'cuda'
_C.distributed = False
_C.log_time = 20
_C.checkpoint_time = 10000
_C.save_dir = 'output'
_C.data_dir = ''  # the annotation folder, e.g. '../datasets/COCO_VLP/annotations'
_C.num_workers = 4
_C.boundaries = ((1, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69))
_C.samples_per_gpu = 64
_C.model_path = ''
_C.pretrained_bert = 'pretrained/bert.pth'
_C.do_noise_eval = True
_C.do_weight_init = True
_C.balanced_training = False

_C.solver = CN()
_C.solver.lr = 5e-5
_C.solver.weight_decay = 1e-2
_C.solver.betas = (0.9, 0.999)
_C.solver.grad_clip = 1.0

_C.scheduler = CN()
_C.scheduler.warmup_steps = 1000
_C.scheduler.max_steps = 300000
_C.scheduler.noisy_training = True
_C.scheduler.noisy_milestone = 100000
_C.scheduler.trusted_milestone = 200000
_C.scheduler.denoise_every = 1200  # eta value from the paper
_C.scheduler.s = 1.0  # smoothness value, s value from the paper
_C.scheduler.p = 2.0  # precentile drop at every stage, c value from the paper
_C.scheduler.full_epochs = False

_C.loss = CN()
_C.loss.balance_weight = 0.5
_C.loss.label_smoothing = 0.1

_C.infer = CN()
_C.infer.steps = (10, 15, 20, 25, 30, 35, 40)
_C.infer.eos_decay = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
