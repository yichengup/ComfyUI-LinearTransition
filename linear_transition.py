import torch
import numpy as np
import comfy.utils
import torch.nn.functional as F

class LinearTransition:
    """
    实现两张图片之间的线性过渡效果，从左到右逐渐过渡
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {"default": "left_to_right"}),  # 过渡方向
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "animation/transition"
    
    def generate_transition(self, image1, image2, frames, direction, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        for i in range(frames):
            # 创建渐变遮罩
            if direction == "left_to_right":
                mask = torch.linspace(0, 1, width).view(1, 1, width).repeat(1, height, 1)
            elif direction == "right_to_left":
                mask = torch.linspace(1, 0, width).view(1, 1, width).repeat(1, height, 1)
            elif direction == "top_to_bottom":
                mask = torch.linspace(0, 1, height).view(1, height, 1).repeat(1, 1, width)
            elif direction == "bottom_to_top":
                mask = torch.linspace(1, 0, height).view(1, height, 1).repeat(1, 1, width)
            
            # 调整遮罩阈值，根据当前帧的位置
            threshold = i / (frames - 1)
            binary_mask = (mask < threshold).float()
            
            # 混合两张图片
            blended = img1 * (1 - binary_mask).unsqueeze(-1) + img2 * binary_mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class GradientTransition:
    """
    实现两张图片之间的平滑渐变过渡效果，使用渐变遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "transition_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),  # 过渡区域宽度
                "direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {"default": "left_to_right"}),  # 过渡方向
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "animation/transition"
    
    def generate_transition(self, image1, image2, frames, transition_width, direction, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        for i in range(frames):
            # 计算当前帧的过渡位置
            position = i / (frames - 1)
            
            # 创建平滑渐变遮罩
            if direction == "left_to_right":
                x = torch.linspace(0, 1, width).view(1, 1, width).repeat(1, height, 1)
                # 创建平滑的sigmoid过渡
                center = position
                mask = 1 / (1 + torch.exp(-(x - center) / (transition_width / 2)))
            elif direction == "right_to_left":
                x = torch.linspace(1, 0, width).view(1, 1, width).repeat(1, height, 1)
                center = 1 - position
                mask = 1 / (1 + torch.exp(-(x - center) / (transition_width / 2)))
            elif direction == "top_to_bottom":
                y = torch.linspace(0, 1, height).view(1, height, 1).repeat(1, 1, width)
                center = position
                mask = 1 / (1 + torch.exp(-(y - center) / (transition_width / 2)))
            elif direction == "bottom_to_top":
                y = torch.linspace(1, 0, height).view(1, height, 1).repeat(1, 1, width)
                center = 1 - position
                mask = 1 / (1 + torch.exp(-(y - center) / (transition_width / 2)))
            
            # 混合两张图片
            blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


# 节点列表，用于注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "LinearTransition": LinearTransition,
    "GradientTransition": GradientTransition
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearTransition": "Linear Transition",
    "GradientTransition": "Gradient Transition"
} 