<h2 align="center"> <a href="https://arxiv.org/abs/2311.10122">Multimodel-LLaVA: Align text with other seven modals base on Languagebind and videollava</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>
<h5 align="center"> <a href="./README_zh.md">中文说明</a></h2>


## 😮 Highlights

We have expanded the capabilities of videollava to eight modalities
- text
- images
- video
- audio
- depth
- thermal
- underwater acoustic
- electromagnetic waves.


## 🛠️ Requirements and Installation
* Python >= 3.10
* Pytorch >= 2.0.1
* CUDA Version >= 11.7
* Install required packages:
```bash
git clone https://github.com/SSSokach/Multimodal_llava.git
cd Video-LLaVA
conda create -n mllava python=3.11
conda activate mllava

# install torch which match your cuda
# show example torch2.1.2+cuda12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"

pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install flash-attn --no-build-isolation
# if flash-attn appear error
# find the verson match your environment, download from https://github.com/Dao-AILab/flash-attention/releases , and install it manually
# pip install /download_path/flash_attn-2.1.0+cu121torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# if torchaudio>2.1.0  maybe appear Error:' No audio backend is available.'
# download backend: sox/soundfile
# conda install -c conda-forge sox
```

## 🗝️ Data &Training & CLI Inference

### data


Prepare your sft data. There are some examples in `./data`.

Here is an image sft data example:

```json
[
    {
        "multimodal_mode": "image"
        "image": "example.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nProvide a one-sentence caption for the provided image.\nReference OCR token: LESS, IN, LESS, IN, SE, TENSE, ZERO, ALHC, ZERO, ALDL, LESS, INTENSE, $S, INTENSE, COHOL, RO, ALCOHOL, ZERO, ALCOHOL, LISTER, ERINE, ZER, LISTER, ZER, LISTERINE, ZERO, STERINE, ERO, MOUTHW, ProventoKill, MOUTHW, ZERO, MOUTH, Millions, Proven, illMillions, Contact, Germs, MOUTHWAS, ermstha, Cause, Breathon, Proven, Kill, Millions, Germs, CLEANMINT, Breath, Contact, CLEANMINT"
            },
            {
                "from": "gpt",
                "value": "Five Listerine Zero mouthwash bottles on a store shelf."
            },
            {
                "from": "human",
                "value": "Thanks."
            },
            {
                "from": "gpt",
                "value": "You're welcome."
            }
        ]
    },
    {
        ...
    },
    {
        ...
    }
]
```

`multimodal_mode`  can be `image，video，audio，depth，thermal，none`, which `none` mean "only text"


### Training

```bash
# prepare model weight
python -m videollava.split_model_utils

# training
bash scripts/v1_5/finetune_continue_sft.sh
```

### CLI Inference 

```bash
bash scripts/v1_5/cli_auto_inferrence.sh
```



## 👍 Acknowledgement

* [LLaVA](https://github.com/haotian-liu/LLaVA) and [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)The codebase we built upon and it is an efficient large language and vision assistant.

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/LICENSE) file.
