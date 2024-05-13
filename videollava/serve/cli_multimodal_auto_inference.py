import argparse
import os

# import debugpy
# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(("localhost", 16823))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
from glob import glob
import torch
import json
from videollava.constants import *
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.serve.utils import load_image, image_ext, video_ext
from videollava.utils import disable_torch_init
from videollava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import random
random.seed(123)
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,indent=2)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit,
                                                                     device=args.device, cache_dir=args.cache_dir)
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    image_start_token = DEFAULT_IMAGE_TOKEN
    video_start_token = DEFAULT_IMAGE_TOKEN * 8
    audio_start_token = DEFAULT_AUDIO_TOKEN
    depth_start_token = DEFAULT_THERMAL_TOKEN
    thermal_start_token = DEFAULT_THERMAL_TOKEN
    electromagnetic_wave_start_token = DEFAULT_ELECTROMAGNETIC_WAVE_TOKEN
    if getattr(model.config, "mm_use_im_start_end", False):
        image_start_token = DEFAULT_IMAGE_START_TOKEN + image_start_token + DEFAULT_IMAGE_END_TOKEN
        video_start_token = DEFAULT_VIDEO_START_TOKEN + video_start_token + DEFAULT_VIDEO_END_TOKEN
        audio_start_token = DEFAULT_AUDIO_START_TOKEN + audio_start_token + DEFAULT_AUDIO_END_TOKEN
        depth_start_token = DEFAULT_DEPTH_START_TOKEN + depth_start_token + DEFAULT_DEPTH_END_TOKEN
        thermal_start_token = DEFAULT_THERMAL_START_TOKEN + thermal_start_token + DEFAULT_THERMAL_END_TOKEN
        electromagnetic_wave_start_token = DEFAULT_ELECTROMAGNETIC_WAVE_START_TOKEN + electromagnetic_wave_start_token + DEFAULT_ELECTROMAGNETIC_WAVE_END_TOKEN

    start_token_dict={
        'image': image_start_token,
        'video': video_start_token,
        'audio': audio_start_token,
        'depth': depth_start_token,
        'thermal': thermal_start_token,
        'electromagnetic_wave': electromagnetic_wave_start_token,
    }
    multimodal_mode_index_map={
        'image': 1,
        'video': 2,
        'audio': 3,
        'depth': 4,
        'thermal': 5,
        'electromagnetic_wave': 6,
    }

        
    eval_data=[]
    for modal in [
        'text',
        'image',
        'video',
        'audio',
        'depth',
        'thermal',
        'seasound',
        'electromagnetic_wave',
        ]:
        json_list=glob(f'/data/home/shilongwang/workplace/A_other_ssd_space/data1/{modal}/universal/*.json')
        for one_file in json_list:
            print(one_file)
            data=load_json(one_file)
            data=random.sample(data, 100)
            for line in tqdm(data):
                eval_data.append(line)
                multimodal_mode=modal
                if multimodal_mode=='text':
                    multimodal_mode='image'
                    file =''
                else:
                    if multimodal_mode=='seasound':
                        multimodal_mode='audio'
                    file=f'/data/home/shilongwang/workplace/A_other_ssd_space/data1/{modal}/universal/{line[multimodal_mode]}'

                conv = conv_templates[args.conv_mode].copy()
                if "mpt" in model_name.lower():
                    roles = ('user', 'assistant')
                else:
                    roles = conv.roles

                if True:
                    if multimodal_mode=='image':
                        image_processor = processor['image']
                        image_file = [file]
                        try:
                            image = [Image.open(file).convert('RGB') for file in image_file]
                            image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                            multimodal_embedding = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                        except Exception as e:
                            print(f'load image file error: ',e)
                            multimodal_embedding = [torch.zeros((3, 224, 224), dtype=torch.float16)]

                    elif multimodal_mode=='video':
                        video_processor = processor['video']
                        video = [file]
                        try:
                            multimodal_embedding = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
                        except Exception as e:
                            print(f'load video file error: ',e)
                            multimodal_embedding = [torch.zeros((3, 8, 224, 224), dtype=torch.float16) for i in video]

                    elif multimodal_mode=='audio':
                        audio_processor = processor['audio']
                        audio = [file]
                        try:
                            multimodal_embedding = [audio_processor(i, return_tensors='pt')['pixel_values'][0] for i in audio]
                        except Exception as e:
                            print(f'load audio file error: ',e)
                            multimodal_embedding = [torch.zeros((3, 126, 1036), dtype=torch.float16) for i in audio]

                    elif multimodal_mode=='depth':
                        depth_processor = processor['depth']
                        depth = [file]
                        try:
                            multimodal_embedding = [depth_processor(i, return_tensors='pt')['pixel_values'][0] for i in depth]
                        except Exception as e:
                            print(f'load depth file error: ',e)
                            multimodal_embedding = [torch.zeros((3, 224, 224), dtype=torch.float16) for i in depth]

                    elif multimodal_mode=='thermal':
                        thermal_processor = processor['thermal']
                        thermal = [file]
                        try:
                            multimodal_embedding = [thermal_processor(i, return_tensors='pt')['pixel_values'][0] for i in thermal]  # fake image
                        except Exception as e:
                            print(f'load thermal file error: ',e)
                            multimodal_embedding = [torch.zeros((3, 224, 224), dtype=torch.float16) for i in thermal]

                    elif multimodal_mode=='electromagnetic_wave':
                        with open(file, 'r', encoding='utf-8') as f:
                            multimodal_embedding = json.load(f)
                        multimodal_embedding = multimodal_embedding + [0] * (1024-len(multimodal_embedding))
                        multimodal_embedding = [torch.tensor([multimodal_embedding], dtype=torch.float16)]
                    # ==========================================================================================================
                    multimodal_embedding = [one.to(model.device, dtype=torch.float16) for one in multimodal_embedding]
                    # ==========================================================================================================
                    
                inp = line['conversations'][0]['value'].replace(f'<{multimodal_mode}>\n', '')
                line['input']=inp
                if file is not None:
                    # first message
                    inp = start_token_dict[multimodal_mode] + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                    file = None
                else:
                    # later messages
                    conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                sep = conv.sep + conv.roles[1] + ": "
                input_ids=[tokenizer.bos_token_id]
                rounds = prompt.split(conv.sep2)
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        part0_ids=tokenizer_image_token(rou, tokenizer)[1:]
                        input_ids+=part0_ids
                        break
                    parts[0] += sep
                    parts[1] += conv.sep2

                    part0_ids=tokenizer_image_token(parts[0], tokenizer)[1:]
                    part1_ids=tokenizer_image_token(parts[1], tokenizer)[1:]

                    input_ids+=part0_ids
                    input_ids+=part1_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=multimodal_embedding,  # video as fake images
                        multimodal_modes=torch.tensor([multimodal_mode_index_map[multimodal_mode]]).cuda(),
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace('</s>','')
                conv.messages[-1][-1] = outputs
                line['output']=outputs
            write_json('/data/home/shilongwang/workplace/Video-LLaVA/assets/test_file/test_result.json', eval_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
