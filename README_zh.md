<h2 align="center"> <a href="https://arxiv.org/abs/2311.10122">Multimodel-LLaVA: 基于Video-LLaVA与LanguageBind对齐文本与七个模态</a></h2>
<h5 align="center"> 如果你喜欢我们的项目，请给我们一个⭐</h2>



## 😮 亮点

我们将videollava的能力扩展到八种模式
- 文本
- 图片
- 视频
- 音频
- 深度
- 热
- 水声
- 电磁波


## 🛠️ 环境配置
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

## 🗝️ 数据 & 训练 & 推理

### 数据


请准备自己的指令微调数据。目录`./data`里提供了一些参考样例。

下面是一个图片模态指令数据的样例:

```json
[
    {
        "multimodal_mode": "image",
        "image": "data_file/image.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n请详细描述给定的图片。"
            },
            {
                "from": "gpt",
                "value": "这张照片捕捉到一名滑板运动员在表演特技时跳上金属栏杆的动作。现场还有其他人。几辆车包围了滑板手练习技巧的区域。可以看到五辆车停在他身后。可以看到第二个滑板被右侧的人骑着。"
            }
        ]
    }
    {
        ...
    },
    {
        ...
    }
]
```

- `multimodal_mode`  可以填 `image，video，audio，depth，thermal，none`,  `none` 代表该条数据是文本数据。
- `image` 填指令对应的图片文件地址。
- `conversations` 支持单轮对话与多轮对话。

### 训练

```bash
# prepare model weight
python -m videollava.split_model_utils

# training
bash scripts/v1_5/finetune_continue_sft.sh
```

### 推理

```bash
bash scripts/v1_5/cli_auto_inferrence.sh
```



## 👍 致谢

* 我们基于[LLaVA](https://github.com/haotian-liu/LLaVA) 与 [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) 这两个代码库实现的，这是一系列优秀的多模态大语言模型。

## 🔒 许可
- 这个项目的大部分是在[license](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/LICENSE)文件中找到的Apache 2.0许可证下发布的。