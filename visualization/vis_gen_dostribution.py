import os
import re

from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from moviepy.editor import ImageSequenceClip

torch.manual_seed = 21



def extract_floats_from_string(s):
    """use regex to find float values after 'x_' and 'y_' """
    match = re.search(r'x_([-+]?\d*\.\d+|\d+)_y_([-+]?\d*\.\d+|\d+)', s)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return x, y

def normalize(value, min_val=-1, max_val=1, grid_min=0, grid_max=None):
    return int((value - min_val) / (max_val - min_val) * (grid_max - grid_min))

def create_large_image(image_paths, coords, grid_size, output_path):
    img_width, img_height = Image.open(image_paths[0]).size
    large_image = Image.new('RGB', (img_width * grid_size[0], img_height * grid_size[1]))

    for path, (x, y) in zip(image_paths, coords):
        grid_x = normalize(x, min_val=-1, max_val=1, grid_min=0, grid_max=grid_size[0]-1)
        grid_y = normalize(y, min_val=-1, max_val=1, grid_min=0, grid_max=grid_size[1]-1)
        small_img = Image.open(path)
        large_image.paste(small_img, (grid_x * img_width, grid_y * img_height))
    large_image.save(output_path)


def create_large_cropped_gif(gif_paths, xy_coords, output_path, grid_size, third=1, margin=5):
    gifs = [Image.open(gif_path) for gif_path in gif_paths]
    width, height = gifs[0].size
    third_width = width // 3 
    
    if third == 1:
        crop_box = (0, 0, third_width-3, height)  # Left third
    elif third == 2:
        crop_box = (third_width, 0, 2 * third_width-4, height)  # Middle third
    elif third == 3:
        crop_box = (2 * third_width, 0, width, height)  # Right third
    else:
        raise ValueError("Invalid value for third. Must be 1, 2, or 3.")
    
    grid_width, grid_height = grid_size
    large_image_width = grid_width * (third_width + margin) - margin
    large_image_height = grid_height * (height + margin) - margin
    large_frames = []
    
    for frame_index, frame in enumerate(ImageSequence.Iterator(gifs[0])):
        large_frame = Image.new("RGBA", (large_image_width, large_image_height), (255, 255, 255, 0))
        for gif, (x, y) in zip(gifs, xy_coords):
            gif.seek(frame_index)
            cropped_gif = gif.crop(crop_box)
            x = int((x + 1) * grid_width // 2) * (third_width + margin)
            y = int((y + 1) * grid_height // 2) * (height + margin)
            large_frame.paste(cropped_gif, (x, y))
        large_frames.append(large_frame)
    large_frames[0].save(output_path, save_all=True, append_images=large_frames[1:], loop=0, duration=gifs[0].info['duration'])


def add_axes_and_title_to_image(image_path, output_path, title="Image with Axes"):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.imshow(img, extent=[-1, 1, -1, 1], aspect='auto', zorder=-1)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    plt.savefig(output_path)
    plt.close()


def add_axes_and_title_to_gif(gif_path, output_path, title="GIF with Axes"):
    gif = Image.open(gif_path)
    new_frames = []
    for frame in ImageSequence.Iterator(gif):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        frame_image = frame.convert('RGB')
        ax.imshow(frame_image, extent=[-1, 1, -1, 1], aspect='auto', zorder=-1)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        fig.canvas.draw()
        new_frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        new_frames.append(new_frame)
        plt.close(fig)

    new_frames[0].save(output_path, save_all=True, append_images=new_frames[1:], loop=0, duration=gif.info['duration'])


def create_video_from_images(image_paths, output_path, fps=10):
    clip = ImageSequenceClip(image_paths, fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps)


vis_imgs=False
if vis_imgs:
    imgs_dir = "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240922_170624/2/infer/2dims_and0_grid"
    for dim in ("x", "y", "z"):
        image_paths=[]
        coords=[]
        for filename in os.listdir(imgs_dir):
            if f"flow_section_{dim}_" in filename and ".png" in filename:
                image_paths.append(os.path.join(imgs_dir, filename))
                coords.append(extract_floats_from_string(filename))
        create_large_image(image_paths, coords, grid_size=(8,8), output_path=f'output_large_image_{dim}.png')
        add_axes_and_title_to_image(f'output_large_image_{dim}.png', f'output_large_image_{dim}_axes.png',title="Grid Sampling in Latent Space: Flow visualization Across Varying z-Values")

vis_gifs=False
if vis_gifs:
    imgs_dir = "/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240922_170624/2/infer/2dims_and0_grid/vis_results"
    gif_paths=[]
    coords=[]
    for filename in os.listdir(imgs_dir):
        if f"generated" in filename and ".gif" in filename:
            gif_paths.append(os.path.join(imgs_dir, filename))
            coords.append(extract_floats_from_string(filename))
    create_large_cropped_gif(gif_paths, coords, output_path=f'output_large_gif_1.gif', grid_size=(8,8), third=1)
    create_large_cropped_gif(gif_paths, coords, output_path=f'output_large_gif_2.gif', grid_size=(8,8), third=2)
    create_large_cropped_gif(gif_paths, coords, output_path=f'output_large_gif_3.gif', grid_size=(8,8), third=3)
    add_axes_and_title_to_gif(f'output_large_gif_1.gif',f'output_large_gif_1_axes.gif',"Grid Sampling in Latent Space: Deformations Across Varying z-Values")
    add_axes_and_title_to_gif(f'output_large_gif_2.gif',f'output_large_gif_2_axes.gif',"Grid Sampling in Latent Space: Deformations Across Varying z-Values")
    add_axes_and_title_to_gif(f'output_large_gif_3.gif',f'output_large_gif_3_axes.gif',"Grid Sampling in Latent Space: Deformations Across Varying z-Values")

vis_flow_change=False
if vis_flow_change:
    imgs_dir="/home/shahar/projects/CVAE_proj/CVAE/outputs/len_44_dataset/20240922_170624/2/infer/2dims_and0_grid"
    img_names = [
        "flow_section_z_x_-1.000_y_-1.000.png",
        "flow_section_z_x_-0.778_y_-0.778.png",
        "flow_section_z_x_-0.556_y_-0.556.png",
        "flow_section_z_x_-0.333_y_-0.333.png",
        "flow_section_z_x_-0.111_y_-0.111.png",
        "flow_section_z_x_0.111_y_0.111.png",
        "flow_section_z_x_0.333_y_0.333.png",
        "flow_section_z_x_0.556_y_0.556.png",
        "flow_section_z_x_0.778_y_0.778.png",
        "flow_section_z_x_1.000_y_1.000.png",
        ]
    image_paths = [os.path.join(imgs_dir, img_name) for img_name in img_names]
create_video_from_images(image_paths, "change_in_flow.mp4")

print("Finished")

