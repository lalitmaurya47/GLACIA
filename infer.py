from segment_model.model_vlm import build_llm_seg
import torch
from torch.cuda.amp import autocast
from PIL import Image
from data_utils.utils import load_image, load_image_rgb
from torchvision import transforms
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token
import cv2
import numpy as np
import os
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
def load_model():
    model, tokenizer, image_processor, config = build_llm_seg(
        model_path="/users/lalit47/GLACIA/weight/llava_mistral_7b", #GLACIA/weight/llava_mistral_7b/",
        model_base=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda:0"
    )
    # model = model.to("cpu")
    tokenizer = model.load_model("/users/lalit47/GLACIA/training_results/train_sam_med_llava_med_new_fixed/best_model_epoch_dice/")
    model = model.to("cuda:0")
    return model, tokenizer, image_processor, config

def transform_for_sam(image_path):
    image_pil = load_image(image_path, channel_indices=[0, 1, 2, 3, 4, 5])       #### Selected Band for prithvi  # channel_indices=[0, 1, 2, 3, 6, 7]
    #image_sam_tensor = self.image_sam_transform(image_pil)
    
    # Convert to torch tensor (C, H, W)
    image_tensor = torch.from_numpy(image_pil).permute(2, 0, 1).float()
    # Resize to 1024x1024 using bilinear interpolation
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
    image_tensor = image_tensor.squeeze(0)  # (C, 1024, 1024)
    
    return image_tensor.unsqueeze(0) #image_sam_tensor.to(torch.float32)

def load_image_for_vlm(image_path, image_processor, config):
    image_pil = load_image_rgb(image_path, channel_indices=[2, 1, 0])        #### R, G, B band 
    plt.imshow(image_pil)
    plt.show()
    image_tensor = process_images(
        [image_pil],
        image_processor,
        config
    )
    return image_tensor.to(torch.float16)

def process_prompt(prompt, tokenizer):
    template = [
        f"Can you identify the location of the glacier lake in this remote sensing image?",
        f"Please describe the glacier lake position in this remote sensing image",
        f"What is the anatomical location of the glacier lake in this remote sensing image?",
        f"Based on this remote sensing image, can you provide the location of the glacier lake in this image?",
        f"Where is the glacier lake located in this remote sensing image?",
    ]
    #prompt_for_vlm = "<image>\n" + f"### User: {random.choice(template)} \n"
    prompt_for_vlm = "<image>\n" + f"### User: {prompt} \n"     #comment by lM
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        -200,
        return_tensors="pt"
    )
    return input_ids.to(torch.int64).unsqueeze(0)

def process_prompt_seg(prompt, tokenizer):
    template = [
        f"Can you identify the location of the glacier lake in this remote sensing image?",
        f"Please describe the glacier lake position in this remote sensing image",
        f"What is the anatomical location of the glacier lake in this remote sensing image?",
        f"Based on this remote sensing image, can you provide the location of the glacier lake in this image?",
        f"Where is the glacier lake located in this remote sensing image?",
    ]
    # question = str(random.choice(template))
    # prompt_for_vlm = "<image> \n" + question
    prompt_for_vlm = "<image> \n" + prompt
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        -200,
        return_tensors="pt"
    )
    return input_ids.to(torch.int64).unsqueeze(0)

def infer(
    prompt,
    image_path,
    image_processor,
    model,
    tokenizer,
    config,
    device = "cuda:0"
):
    image_tensor = load_image_for_vlm(image_path, image_processor, config)
    #image_tensor = transform_for_sam(image_path)                       ######Added by me 
    image_tensor_for_sam = transform_for_sam(image_path)
    image_tensor_for_sam = image_tensor_for_sam.to(device)
    image_tensor = image_tensor.to(device)
    input_ids = process_prompt(prompt, tokenizer)
    input_ids_for_seg = process_prompt_seg(prompt, tokenizer).to(device)
    input_ids = input_ids.to(device)
    # print(input_ids.shape)
    # print(image_tensor.shape)
    # print(image_tensor.dtype)
    #print(input_ids)
    model.eval()
    model.to(device)
    # threshold = 0.5
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            output_mask, output_ids = model.generate(
                input_ids = input_ids,
                input_ids_for_seg = input_ids_for_seg,
                image_tensor_for_vlm = image_tensor,
                image_tensor_for_image_enc = image_tensor_for_sam,
                attention_mask = None,
                temperature=0.2,
                max_new_tokens=512,
                top_p=0.95
            )
            #print(output_mask.shape, output_ids.shape)
            #print(output_ids)
            #print(output_ids[0, input_ids.shape[1]:])
            #print(output_ids[0, :])
            # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            outputs_1 = tokenizer.decode(output_ids[0, :], skip_special_tokens=True)
            # print("Output:", outputs)
            # print("Output 1:", outputs_1)
            res = output_mask.sigmoid().cpu().numpy().squeeze()
            # res = (res > 0).astype(np.uint8)
            #print(res.shape)
            # res = masks.transpose(1, 2, 0)
            # res = masks
            #print(res.min(), res.max())
            #print(res.shape)
            #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # res = (res > 0.3).astype(np.uint8)
            # cv2.imwrite("output_mask.png", res* 255)
            return res, outputs_1
            
    
            # outputs_answers = tokenizer.batch_decode(
            #     output_ids[:, input_ids.shape[1]:],
            #     skip_special_tokens=True
            # )[0]
            # print(outputs_answers)


import pandas as pd
model, tokenizer, image_processor, config = load_model()

results_path = "results/lm_seg_test_2_new_data"

mask_path = results_path + "/masks/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
    os.makedirs(mask_path)
# df_res = pd.DataFrame(columns=["image_path", "mask_path", "prompt", "results"])
image_list = []
mask_list = []
prompt_list = []
answer_list = []
df = pd.read_csv("/users/lalit47/GLACIA/annotation/glacier_dataset.csv")
# len(df)
# cnt =0 
for i in range(len(df)):
    # if cnt > 10:
        # break
    if df.iloc[i]["split"] == "test":
        # cnt+=1
        # beo
        #image_path = "/home/mamba/ML_project/Testing/Huy/llm_seg/dataset/data/" + df.iloc[i]["image_path"]
        image_path = df.iloc[i]["image_path"]
        image_path = image_path.replace("test/masks", "test/images").replace(".png", ".tif")  
        #print("image_path", image_path)
        idx = df.iloc[i]["image_path"].split("/")[-1].split(".")[0]
        modal = df.iloc[i]["image_path"].split("/")[0]
        prompt = df.iloc[i]["question"]
        print("Image path:", image_path)
        print("Prompt:", prompt)
        mask, answer = infer(
            prompt,
            image_path,
            image_processor,
            model,
            tokenizer,
            config
        )
        mask = mask * 255
        #mask = mask.astype(np.uint8)
        print(mask.max())
        if not os.path.exists(mask_path + "/" + modal):
            os.makedirs(mask_path + "/" + modal)
        save_mask_path = mask_path + f"/{modal}/" + str(idx) + ".png"
        cv2.imwrite(save_mask_path, mask)
        plt.imshow(mask)
        plt.show()
        print("Save mask path:", save_mask_path, "| Answer:", answer)
        image_list.append(df.iloc[i]["image_path"])
        mask_list.append(save_mask_path)
        prompt_list.append(prompt)
        answer_list.append(answer.replace("\n","").replace("### Assistant: ","").replace("### User: ","").replace("You are doing the segmentation for the glacier with the condition: ",""))
        # df_res = pd.concat([df_res, pd.DataFrame({"image_path": [df.iloc[i]["image_path"]], "mask_path": [save_mask_path], "prompt": [prompt], "results": [answer]})], ignore_index=True)
df_res = pd.DataFrame({
    "image_path": image_list,
    "mask_path": mask_list,
    "prompt": prompt_list,
    "results": answer_list
})
print("Number of answer:", len(answer_list), "Number of image:", len(image_list), "Number of mask:", len(mask_list))
df_res.to_csv(results_path + "/results.csv", index=False, header = False)

    # infer(
    #     prompt,
    #     image_path,
    #     image_processor,
    #     model,
    #     tokenizer,
    #     config
    # )


# model.to("cuda:1")
