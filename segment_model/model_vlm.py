import requests
from PIL import Image
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch.nn as nn
import torch 
from segment_model.mask_decoder_vlm import PromptedMaskDecoder
import peft
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel
import math 


import torch.nn.functional as F
from terratorch.registry import BACKBONE_REGISTRY
from torchvision.models import resnet18
from segment_model.prithvi_res import HybridPrithviEncoder



def custom_lora_init(module):
    if hasattr(module, "lora_A"):
        nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
    if hasattr(module, "lora_B"):
        nn.init.zeros_(module.lora_B.weight)


class LLMSeg(nn.Module):
    def __init__(
            self, 
            model_path, 
            model_base=None, 
            load_8bit=False, 
            load_4bit=False, 
            device="cuda:0"
        ):

        super(LLMSeg, self).__init__()
        disable_torch_init()
        self.device = device        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=self.device
        )
        
        

        print("✅ PrithviRes successfully attached.")
        ###################################

        
        self.base_model.eval()

        self.model = get_peft_model(self.base_model, lora_config)
        # self.model.to(dtype=torch.float32)
        if self.training:
            self.model.to(dtype=torch.bfloat16)
        # else:
        #     self.model.to(dtype=torch.float32)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # self.model = self.model.to_fp32()

        self.mask_decoder = PromptedMaskDecoder()

        self.image_encoder = HybridPrithviEncoder(
            model_name="prithvi_eo_v2_100_tl",
            checkpoint_path= "/users/lalit47/GLACIA/weight/prithvi/Prithvi_EO_V2_100M_TL.pt",
            use_tower=False
        )
       
        self.image_encoder.train()
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )
        torch.nn.init.xavier_uniform_(self.cls[2].weight)
        torch.nn.init.ones_(self.cls[2].bias)

    
        

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.base_model.config
    
    def save_model(self, save_path):
        self.model.save_pretrained(save_path + "/lora_adapter")
        self.tokenizer.save_pretrained(save_path + "/tokenizer")
         # Save CLIP vision tower if you trained it
        #torch.save(self.base_model.get_model().vision_tower.state_dict(), save_path + "/vision_tower.pth")
        #torch.save(self.base_model.get_model().mm_projector.state_dict(), save_path + "/mm_projector.pth")


        torch.save(self.image_encoder.state_dict(), save_path + "/image_encoder.pth")
        torch.save(self.mask_decoder.state_dict(), save_path + "/mask_decoder.pth")
        torch.save(self.cls.state_dict(), save_path + "/cls.pth")

    def load_model(self, load_path):
        print("Loading model from:", load_path)
        self.tokenizer = self.tokenizer.from_pretrained(load_path + "/tokenizer/")
        self.mask_decoder.load_state_dict(torch.load(load_path + "/mask_decoder.pth"))
        self.image_encoder.load_state_dict(torch.load(load_path + "/image_encoder.pth"))
        # Load CLIP vision tower weights
        #self.base_model.get_model().vision_tower.load_state_dict(torch.load(load_path + "/vision_tower.pth"))
        #self.base_model.get_model().mm_projector.load_state_dict(torch.load(load_path + "/mm_projector.pth"))

        self.model = PeftModel.from_pretrained(self.model, load_path + "/lora_adapter/")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # self.cls.load_state_dict(torch.load(load_path + "/cls.pth"))
        # self.model.generation_config.IMAGE_TOKEN_ID = self.tokenizer.IMAGE_TOKEN_ID
        self.mask_decoder.to(self.device)
        self.mask_decoder.eval()
        self.model = self.model.merge_and_unload()
        self.image_encoder.eval()
        self.model.eval()
        return self.tokenizer
    
    def generate(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        input_ids_for_seg=None,
        attention_mask = None,
        temperature=0.1,
        max_new_tokens=512,
        top_p=0.95
    ):
        self.image_encoder.eval()
        self.model.eval()
        self.mask_decoder.eval()
        
       
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs = input_ids,
                images = image_tensor_for_vlm,
                #image_sizes=image_sizes,                 ######added by lm 
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )

            image_embedding = self.image_encoder(image_tensor_for_image_enc)
            
            #print("image_embedding", image_embedding.shape)
            # shape_embedding = input_ids.shape[-1]
            # prompt_embedding = self.model.extract_last_hidden_state(
            #     input_ids = input_ids,
            #     images = image_tensor_for_vlm,
            #     do_sample=False,
            #     temperature=0,
            #     max_new_tokens=max_new_tokens,
            #     top_p=top_p
            # # )["hidden_states"][-1][:shape_embedding, :]
             ###############################
            # Encode image with Prithvi
            # vision_feats = self.base_model.get_model().vision_tower(image_tensor_for_vlm)  # [B, 256]
            # vision_feats = self.vision_bridge(vision_feats)  # [B, 1024]
            
            prompt_embedding = self.base_model.extract_last_hidden_state(
                input_ids = input_ids_for_seg if input_ids_for_seg is not None else input_ids,
                images = image_tensor_for_vlm,
                #image_sizes=image_sizes,                 ######added by lm 
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
            final_mask = self.mask_decoder(
                image_embedding, prompt_embedding
            )
        return final_mask, output_ids

    def forward(self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        attention_mask = None,
        answers=None,
        temperature=0.0001,
        max_new_tokens=512,
        top_p=0.95
    ):
        if self.training:
            self.model.to(dtype=torch.bfloat16)
        else:
            self.model.to(dtype=torch.float16)

        # with torch.no_grad():
            # prompt_embedding = self.model.extract_last_hidden_state(
            #     input_ids = input_ids,
            #     images = image_tensor_for_vlm,
            #     do_sample=False,
            #     temperature=0,
            #     max_new_tokens=max_new_tokens,
            #     top_p=top_p
            # )["hidden_states"][-1]
        #image_sizes = [(224, 224)] * image_tensor_for_vlm.size(0)
#################################################################################
        # 🔵 Mask out prompt tokens so loss is computed only for answer part
        labels = answers.clone()
        # prompt_length = input_ids.shape[1]
        # labels[:, :prompt_length] = -100
#################################################################################


        ###############################
        # Encode image with Prithvi
        # print(image_tensor_for_vlm.shape)
        # image_tensor_for_vlm = image_tensor_for_vlm.unsqueeze(2)
        # print(image_tensor_for_vlm.shape)
        # vision_feats = self.base_model.get_model().vision_tower(image_tensor_for_vlm)  # [B, 256]
        # vision_feats = self.vision_bridge(vision_feats)  # [B, 1024]
###########################################
        prompt_embedding = self.model(
            input_ids = answers,
            attention_mask=attention_mask,
            images= image_tensor_for_vlm,
            #image_sizes=image_sizes,                 ######added by lm 
            use_cache = False,
            labels=labels,                           ###### answers
            return_dict=True, 
            output_hidden_states=True,
        )["hidden_states"][-1]

        image_embedding = self.image_encoder(image_tensor_for_image_enc)
        #image_embedding = image_embedding_list[-1]
        #print("image_embedding", image_embedding.shape)
        #output_cls = self.cls(image_embedding)
        output_cls = 0
        # print("Output cls:", output_cls)
        image_embedding = image_embedding.to(self.device)
        prompt_embedding = prompt_embedding.to(self.device)
        # print("image_embedding", image_embedding.shape)
        # print("prompt_embedding", prompt_embedding.shape)
        final_mask = self.mask_decoder(
            image_embedding, prompt_embedding
        )
        if self.training:
            logit_loss = self.model(
                input_ids = answers,                            #answers 
                attention_mask=attention_mask,
                images= image_tensor_for_vlm,
                #image_sizes=image_sizes,                 ######added by lm 
                use_cache = False,
                labels=labels
            ).loss
            return final_mask, output_cls, logit_loss
        else:
            output = self.model(
                input_ids = answers,
                attention_mask=attention_mask,
                images= image_tensor_for_vlm,
                #image_sizes=image_sizes,                 ######added by lm 
                use_cache = False,
                labels=labels
            ).logits
            return final_mask, output

def build_llm_seg(
        model_path, 
        model_base=None, 
        load_8bit=False, 
        load_4bit=False, 
        device="cuda:0"
):
    llm_seg = LLMSeg(
        model_path=model_path,
        model_base=model_base,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device
    )

    tokenizer, image_processor, context_len, config = llm_seg.get_model_utils()
    return llm_seg, tokenizer, image_processor, config