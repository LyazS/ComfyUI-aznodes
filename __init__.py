import numpy as np
import os
import folder_paths
from PIL import Image


class CrossFadeImageSequence:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crossfadeimagesequence"
    CATEGORY = "AZNodes"
    DESCRIPTION = (
        "Creates a smooth transition between two image sequences by crossfading."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_1": ("IMAGE",),
                "images_2": ("IMAGE",),
                "interpolation": (
                    [
                        "linear",
                        "ease_in",
                        "ease_out",
                        "ease_in_out",
                        "bounce",
                        "elastic",
                        "glitchy",
                        "exponential_ease_out",
                    ],
                ),
                "transitioning_frames": (
                    "INT",
                    {"default": 1, "min": 0, "max": 4096, "step": 1},
                ),
                "start_level": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_level": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    def crossfadeimagesequence(
        self,
        images_1,
        images_2,
        transitioning_frames,
        interpolation,
        start_level,
        end_level,
    ):
        import math
        import torch

        def crossfade(images_1, images_2, alpha):
            crossfade = (1 - alpha) * images_1 + alpha * images_2
            return crossfade

        def ease_in(t):
            return t * t

        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t

        def bounce(t):
            if t < 0.5:
                return ease_out(t * 2) * 0.5
            else:
                return ease_in((t - 0.5) * 2) * 0.5 + 0.5

        def elastic(t):
            return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))

        def glitchy(t):
            return t + 0.1 * math.sin(40 * t)

        def exponential_ease_out(t):
            return 1 - (1 - t) ** 4

        easing_functions = {
            "linear": lambda t: t,
            "ease_in": ease_in,
            "ease_out": ease_out,
            "ease_in_out": ease_in_out,
            "bounce": bounce,
            "elastic": elastic,
            "glitchy": glitchy,
            "exponential_ease_out": exponential_ease_out,
        }

        # 获取两组图像的帧数
        frames_1 = len(images_1)
        frames_2 = len(images_2)

        # 检查序列长度
        if frames_1 == 0 or frames_2 == 0:
            raise ValueError("input image sequence is empty")

        # 确保过渡帧数不超过任一序列的帧数
        if transitioning_frames > frames_1 or transitioning_frames > frames_2:
            # print(f"警告: 过渡帧数 {transitioning_frames} 超过了序列长度 (序列1: {frames_1}, 序列2: {frames_2})。")
            # print(f"自动调整过渡帧数为 {min(frames_1, frames_2)}")
            raise ValueError(
                f"transitioning frames is too large than input image sequence length: {frames_1} and {frames_2}"
            )

        # 如果过渡帧数为0，则直接连接两个序列
        if transitioning_frames == 0:
            return (torch.cat([images_1, images_2], dim=0),)

        # 准备结果图像列表
        result_images = []

        # 添加序列A中不需要过渡的前部分帧
        non_transition_frames_1 = frames_1 - transitioning_frames
        if non_transition_frames_1 > 0:
            result_images.append(images_1[:non_transition_frames_1])

        # 处理过渡帧
        alphas = torch.linspace(start_level, end_level, transitioning_frames)
        easing_function = easing_functions.get(interpolation)

        crossfade_images = []
        for i in range(transitioning_frames):
            alpha = alphas[i]
            alpha = easing_function(alpha)  # 应用缓动函数

            image1 = images_1[non_transition_frames_1 + i]
            image2 = images_2[i]

            crossfade_image = crossfade(image1, image2, alpha)
            crossfade_images.append(crossfade_image)

        if crossfade_images:
            crossfade_tensor = torch.stack(crossfade_images, dim=0)
            result_images.append(crossfade_tensor)

        # 添加序列B中不需要过渡的后部分帧
        if frames_2 > transitioning_frames:
            result_images.append(images_2[transitioning_frames:])

        # 合并所有结果
        if len(result_images) > 1:
            final_result = torch.cat(result_images, dim=0)
        else:
            final_result = result_images[0]

        return (final_result,)


class SaveImageAZ:
    def __init__(self):
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": "The prefix for the file to save. Includes the folder name to save to specify the subfolder.",
                    },
                ),
                "output_folder": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": "The folder to save the images to. If it is not abspath, use default ComfyUI output folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "AZNodes/image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, output_folder, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append

        if os.path.isabs(output_folder):
            full_output_folder = output_folder
            _, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    output_folder,
                    images[0].shape[1],
                    images[0].shape[0],
                )
            )
        else:
            self.output_dir = folder_paths.get_output_directory()
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    self.output_dir,
                    images[0].shape[1],
                    images[0].shape[0],
                )
            )
            print(full_output_folder)
        # 检查并重新创建输出文件夹
        if os.path.exists(full_output_folder):
            import shutil

            shutil.rmtree(full_output_folder)
        os.makedirs(full_output_folder, exist_ok=True)

        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            base_file_name = f"{filename_with_batch_num}_{counter:05}"
            file = f"{base_file_name}.png"
            img.save(
                os.path.join(full_output_folder, file),
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )

            counter += 1

        return (file,)


NODE_CLASS_MAPPINGS = {
    "CrossFadeImageSequence": CrossFadeImageSequence,
    "SaveImageAZ": SaveImageAZ,
}
