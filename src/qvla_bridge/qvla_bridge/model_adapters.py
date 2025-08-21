#!/usr/bin/env python3
"""
Model adapter layer
"""
import requests
import cv2
import numpy as np
from PIL import Image
from enum import Enum

class ModelType(Enum):
    OPENVLA = "openvla"
    NAVILA = "navila"

# API configuration
OPENVLA_API = "http://127.0.0.1:8000/act"

def call_openvla_api(image, instruction, timeout=15.0):
    """
    Call OpenVLA API
    
    Args:
        image: PIL Image or numpy array
        instruction: instruction text
        timeout: timeout duration
    
    Returns:
        API response JSON data
    """
    # Ensure image is in numpy array format
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to RGB format (if needed)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', img_array)
    import base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Build request
    payload = {
        "image": img_base64,
        "instruction": instruction
    }
    
    try:
        response = requests.post(OPENVLA_API, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"OpenVLA API call failed: {e}")
        return None

def call_navila_inference(image_sequence, instruction):
    """
    Call NaVILA model for inference - Standard GPU inference (32GB VRAM)
    
    Args:
        image_sequence: image sequence list
        instruction: instruction text
    
    Returns:
        model output text
    """
    try:
        # Required imports
        import torch
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import (
            process_images, tokenizer_image_token, get_model_name_from_path
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import KeywordsStoppingCriteria
        from llava.utils import disable_torch_init
        
        # Model path
        model_path = "a8cheng/navila-llama3-8b-8f"
        
        # Check if model is already cached
        if not hasattr(call_navila_inference, '_model_cache'):
            call_navila_inference._model_cache = {}
        
        if model_path not in call_navila_inference._model_cache:
            print(f"Loading NaVILA model from {model_path}")
            
            # GPU memory check
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, please check your GPU")
            
            # Standard model loading (reference run_navigation.py)
            disable_torch_init()
            model_name = get_model_name_from_path(model_path)
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, 
                model_name, 
                None,  # model_base
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            print("✅ NaVILA loaded successfully")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Model precision: {next(model.parameters()).dtype}")
            
            # Cache model
            call_navila_inference._model_cache[model_path] = {
                'tokenizer': tokenizer,
                'model': model,
                'image_processor': image_processor,
                'context_len': context_len
            }
        
        # Get cached model
        cache = call_navila_inference._model_cache[model_path]
        tokenizer = cache['tokenizer']
        model = cache['model']
        image_processor = cache['image_processor']
        
        # Check input
        if len(image_sequence) == 0:
            return "No images provided for navigation"
        
        # Build prompt (consistent with run_navigation.py)
        num_frames = len(image_sequence)
        image_token = "<image>\n"
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_token * (num_frames-1)}, and current observation <image>\n. Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        
        # Conversation template
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process images and text (reference run_navigation.py)
        images_tensor = process_images(image_sequence, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        # Stop criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # Inference (consistent with run_navigation.py)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.half().cuda(),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        
        return {"text": outputs, "prompt": prompt}
        
    except Exception as e:
        print(f"NaVILA inference failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Inference failed: {e}"



def call_model(model_type, model_input, instruction):
    """
    Unified model dispatcher
    Returns model raw output (OpenVLA: vector; NaVILA: text)
    """
    dispatch = {
        ModelType.OPENVLA: lambda: call_openvla_api(model_input, instruction) or np.zeros(7),
        ModelType.NAVILA:  lambda: call_navila_inference(model_input, instruction),
    }
    if model_type not in dispatch:
        raise ValueError(f"Unsupported model type: {model_type}")
    return dispatch[model_type]()