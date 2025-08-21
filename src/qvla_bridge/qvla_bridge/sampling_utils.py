#!/usr/bin/env python3
import numpy as np
from PIL import Image
from collections import deque

def adaptive_frame_sampling(frames, target_frames=8, preserve_current=True):
    """
    Adaptive frame sampling with history-current separation strategy
    
    Sample input frame sequence to fixed length, maintaining temporal and semantic integrity.
    Prioritize current observation frame, use linear interpolation sampling for historical frames.
    
    Args:
        frames: input frame sequence (list of PIL.Image)
        target_frames: target frame count, default 8
        preserve_current: whether to preserve the latest frame as current observation
    
    Returns:
        list: sampled frame sequence
        
    Note:
        Sampling strategy references NaVILA project's get_frame_from_vcap_vlnce_navila_v2 function
        Original code: https://github.com/zzp-seeker/NaVILA/blob/main/llava/mm_utils.py
    """
    if len(frames) == 0:
        print("Warning: Empty frame sequence")
        return []
    
    # Return short sequences directly
    if len(frames) <= target_frames:
        return frames
    
    if preserve_current:
        # Separate current observation and historical frames
        current_frame = frames[-1:]
        history_frames = frames[:-1]
        
        # Resample historical frames
        if len(history_frames) == (target_frames - 1):
            # Perfect match, return directly
            return frames
        else:
            # Linear interpolation sampling for historical frames
            indices = np.linspace(0, len(history_frames) - 1, target_frames - 1)
            sampled_history = []
            
            for idx in indices:
                nearest_idx = int(round(idx))
                sampled_history.append(history_frames[nearest_idx])
            
            return sampled_history + current_frame
    else:
        # Global uniform sampling
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        return [frames[i] for i in indices]

def linear_interpolation_sampling(frames, target_frames=8):
    """
    Linear interpolation sampling strategy, maintaining temporal consistency
    
    Args:
        frames: input frame sequence
        target_frames: target frame count
    
    Returns:
        sampled frame sequence
        
    Note:
        Simplified temporal sampling, suitable for various video processing tasks
    """
    if len(frames) == 0:
        return []
    
    if len(frames) <= target_frames:
        return frames
    
    # Uniform sampling
    indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
    return [frames[i] for i in indices]
