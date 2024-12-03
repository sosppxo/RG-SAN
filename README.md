# RG-SAN: Rule-Guided Spatial Awareness Network for End-to-End 3D Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)

NEWS:🔥RG-SAN is accepted at NeurIPS 2024 (Oral)!🔥

We invite you to explore our series of works, including [3D-STMN (AAAI 2024)](https://github.com/sosppxo/3D-STMN) and [3D-GRES (MM 2024 Oral)](https://github.com/sosppxo/MDIN).

Changli Wu, Qi Chen, Jiayi Ji, Haowei Wang, Yiwei Ma, You Huang, Gen Luo, Hao Fei, Xiaoshuai Sun, Rongrong Ji

[Demo](https://stuxmueducn-my.sharepoint.com/personal/22920182204313_stu_xmu_edu_cn/_layouts/15/stream.aspx?id=%2Fpersonal%2F22920182204313%5Fstu%5Fxmu%5Fedu%5Fcn%2FDocuments%2FRG%2DSAN%2FDemo%5FRG%2DSAN%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E24238020%2Db322%2D4ee0%2D99b8%2D5e531c67ffc5)

Framework:

<img src="docs\RG-SAN.png"/>

## Introduction
3D Referring Expression Segmentation (3D-RES) aims to segment 3D objects by correlating referring expressions with point clouds. However, traditional approaches frequently encounter issues like over-segmentation or mis-segmentation, due to insufficient emphasis on spatial information of instances. In this paper, we introduce a Rule-Guided Spatial Awareness Network  (RG-SAN) by utilizing solely the spatial information of the target instance for supervision. This approach enables the network to accurately depict the spatial relationships among all entities described in the text, thus enhancing the reasoning capabilities. The RG-SAN consists of the Text-driven Localization Module (TLM) and the Rule-guided Weak Supervision (RWS) strategy. The TLM initially locates all mentioned instances and iteratively refines their positional information. %This step ensures that each entity's location is continuously improved. The RWS strategy, acknowledging that only target objects have supervised positional information, employs dependency tree rules to precisely guide the core instance's positioning. Extensive testing on the ScanRefer benchmark has shown that RG-SAN not only establishes new performance benchmarks, with an mIoU increase of 5.1 points, but also exhibits significant improvements in robustness when processing descriptions with spatial ambiguity.

## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n rg-san python=3.8
  conda activate rg-san
  ```

- Clone this repository

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl # please check the versions in the website
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).
  
  Install Stanford CoreNLP toolkit from the [official website](https://stanfordnlp.github.io/CoreNLP/download.html).

- Setup, Install rg_san, pointgroup_ops and attention_rpe_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  # install pointgroup_ops
  cd rg_san/lib/ && python setup.py develop && cd ../../
  # install attention_rpe_ops
  cd lib/attention_rpe_ops && python setup.py install && cd ../../
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows.

```
RG-SAN
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
RG-SAN
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
```

### ScanRefer dataset
Download [ScanRefer](https://github.com/daveredrum/ScanRefer) annotations following the instructions.

Put the downloaded `ScanRefer` folder as follows.
```
RG-SAN
├── data
│   ├── ScanRefer
│   │   ├── ScanRefer_filtered_train.json
│   │   ├── ScanRefer_filtered_val.json
```
Preprocess textual data
```
python data/features/save_graph.py --split train --data_root data/ --max_len 78
python data/features/save_graph.py --split val --data_root data/ --max_len 78
```

## Pretrained Backbone

Download [3D U-Net](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/Em7yJHaCHAxFpM15uVwk9cgByDp-67lWQg59vkU-zokHYA?e=IuZV0D) pretrained weights from [3D-STMN](https://github.com/sosppxo/3D-STMN).

Move the pretrained model to backbones.
```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Training
```
bash scripts/train.sh
```

## Inference
You can download our [checkpoint](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/Eg54iH5XXDFErLI674r_dCEB_GCpHtf9g5U0WWBfvQvtqA?e=0qsifc) to reproduce the performance:
```
bash scripts/test.sh
```
