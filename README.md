# CLID: Controlled-Length Image Descriptions with Limited Data

This repository provides the code for our paper CLID (WACV 2024).

## Installation

### Python environment

```
conda create --name clid python=3.7
conda activate clid

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install h5py tqdm transformers==2.1.1 tensorboardX yacs
pip install git+https://github.com/salaniz/pycocoevalcap
```

### Data

* Prepare MS-COCO data following this [link](https://github.com/LuoweiZhou/VLP#-data-preparation).

* Prepare MS-COCO trusted and noisy annotation files, as in these files (soon).

* Download the pretrained BERT model ([link](https://drive.google.com/file/d/1B3R2wTYeoiXdT4HsKj4EEdWPnKdZjF3L/view?usp=sharing)).

## Running the code

First update _config_denoise.py_ with the correct data and pretrained model paths.

The hierarchy above $ _C.data_dir $ should contain two folders.
The first is region_feat_gvd_wo_bgd (downloaded in the previous section), containing the visual features.
The second is the annotation folder (name defined by $ _C.data_dir $), containing the annotations.

|── Data parent folder\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── region_feat_gvd_wo_bgd (containing feature files)\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|── $ _C.data_dir $ (containing annotations)

Note that we use a yacs configuration file, hence you may modify the config file or use command line arguments.
A command line argument overrides the config file's one.

**Train**

```
python train_denoise.py save_dir checkpoints_noisy samples_per_gpu 64
```

**Continue from checkpoint**

```
python train_denoise.py model_path <path_to_.pth> save_dir checkpoints_noisy samples_per_gpu 64
```

**Inference & evaluate**

You shall set the path to the test captions in _gt_caption_. 
it is the path to a file named _id2captions_test.json_, which 
will be created and located in the folder of the annotation files. 
Hence, set it to $ _C.data_dir $/id2captions_test.json.

In _model_path_, set the model you desire to infer.

```
python infer_and_eval.py \
--gt_caption <path_to_'id2captions_test.json'> \
--pd_caption output/caption_results.json \
--save_dir output \
model_path <path_to_.pth> \
save_dir output \
samples_per_gpu 64
```

## Citing our work

Please consider citing our paper if the project helps your research.

```
@inproceedings{hirsch2024clid,
  title={CLID: Controlled-Length Image Descriptions with Limited Data},
  author={Hirsch, Elad and Tal, Ayellet},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5531--5541},
  year={2024}
}
```

## Acknowledgements

We thank the authors of LaBERT, both for their research and for sharing their code.
Our repository is built upon their [environment](https://github.com/bearcatt/LaBERT).