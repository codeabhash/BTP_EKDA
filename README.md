# **End-to-End Knowledge Distillation for Unsupervised DomainAdaptation with Large Vision-language Models**

------

## Highlights

![Architecture](https://anonymous.4open.science/r/EKDA/Architecture.png)

> **Abstract:** Knowledge distillation based on large vision-language models (VLMs) has recently emerged as a significant solution to transfer knowledge from the source domain to the target domain in unsupervised domain adaptation (UDA) tasks. However, existing methods employ a two-stage training pipeline, which not only complicates the training procedure but also lacks interactions between the source and target domains, severely hindering real-time cross-domain knowledge transfer. To address these challenges, we propose End to End Knowledge Distillation for UDA with large VLMs (termed as EKDA). (1) EKDA employs a lightweight prompt learning mechanism to first embed the knowledge from the source domain into VLMs, and then simultaneously utilize the image encoder and text encoder of VLMs to perform knowledge distillation on the target domain, significantly reducing the domain gap. (2) EKDA designs a teacher-student alternating training strategy to implement realtime collaborative interactions across domains, enabling an end-toend paradigm to provide accurate source domain-aware supervision for the target domain. We conduct extensive experiments on 4 widely recognized benchmark datasets including Office-31, OfficeHome, VisDA-2017, and Mini-DomainNet. Experimental results demonstrate that EKDA achieves significant performance improvement over the state-of-the-art UDA approaches, while maintaining a much lower model complexity. Take Office-Home for example, EKDA has gained at least 2.7% performance improvement while reducing the learnable parameters by over 80% compared with the strongest baselines. 

## Main Contributions



- **New paradigm**： To the best of our knowledge, this is the first attempt to propose an end-to-end knowledge distillation paradigm that facilitates real-time collaborative interactions between the source domain and target domain, breaking new ground for UDA with VLMs-based cross-domain knowledge transfer.
- **Novel method：** We introduce a lightweight cross-domain distillation method EKDA that (1) leverages text features from the source domain as shared semantic centers, and (2) extracts source domain-aware image features and pseudo labels for the target domain to implement real-time knowledge distillation, significantly reducing the domain gap.
- **High Performance：** We conduct extensive experiments on UDA benchmark datasets including Office-31, OfficeHome, VisDA-2017, and MiniDomainNet. Experimental results demonstrate that EKDA achieves significant performance improvement over the state-of-the-art approaches, while maintaining a much lower model complexity



## Installation



- Setup conda environment.

```bash
# Create a conda environment
conda create -y -n ekda python=3.8

# Activate the environment
conda activate ekda

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



- Clone EKDA code repository and install requirements.

```bash
# Clone EKDA  code base
git clone https://github.com/1d1x1w/EKDA.git
cd EKDA

# Install requirements
pip install -r requirements.txt
```



## Data preparation



Please follow the instructions as follows to prepare all datasets. Datasets list:

- [Office-31](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf)
- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
- [VisDA17](http://ai.bu.edu/visda-2017/)
- [DomainNet](http://ai.bu.edu/M3SDA/)
- [miniDomainNet](https://arxiv.org/abs/2003.07325)



## Training and Evaluation

Please follow the instructions for training, evaluating and reproducing the results. Firstly, you need to **modify the directory of data by yourself** in scripts/main_ekda.sh and scripts/eval_ekda.sh.

### Training

```bash
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/ekda/main_ekda.sh officehome b32_ep20_officehome EKDA ViT-B/16 2 a-c 0
```



### Evaluation

```
# evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/ekda/eval_ekda.sh officehome b32_ep20_officehome EKDA ViT-B/16 2 a-c 0
```



The details are at each method folder in [scripts folder]([EKDA/scripts at main · 1d1x1w/EKDA (github.com)](https://anonymous.4open.science/r/EKDA/scripts)).



## Acknowledgements



Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) ，[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) ，[PromptSRC](https://github.com/muzairkhattak/PromptSRC) , [PromptKD](https://github.com/zhengli97/PromptKD) and [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) etc. repository. We thank the authors for releasing their code. If you use their model and code, please consider citing these works as well. Supported methods are as follows:

| Method       | Paper                                          | Code                                                         |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| CoOp         | [IJCV 2022](https://arxiv.org/abs/2109.01134)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| CoCoOp       | [CVPR 2022](https://arxiv.org/abs/2203.05557)  | [link](https://github.com/KaiyangZhou/CoOp)                  |
| VPT          | [ECCV 2022](https://arxiv.org/abs/2203.17274)  | [link](https://github.com/KMnP/vpt)                          |
| IVLP & MaPLe | [CVPR 2023](https://arxiv.org/abs/2210.03117)  | [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| DAPL         | [TNNLS 2023](https://arxiv.org/abs/2202.06687) | [link](https://github.com/LeapLabTHU/DAPrompt)               |
| PromptSRC    | [ICCV2023](https://arxiv.org/abs/2307.06948)   | [link](https://github.com/muzairkhattak/PromptSRC)           |
| PromptKD     | [CVPR2024](https://arxiv.org/abs/2403.02781)   | [link](https://github.com/zhengli97/PromptKD)                |
| PDA          | [AAAI 2024](https://arxiv.org/abs/2312.09553)  | [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment) |



