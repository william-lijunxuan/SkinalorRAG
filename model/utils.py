import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def imageCompress(quality, subsampling, img_item, notfound_imgs, jpga, output):

    img_item_path = ''
    img_item_path = Path(os.path.abspath(img_item))
    img_item_endswith = img_item.endswith('.png') or img_item.endswith('.jpg') or img_item.endswith(
        '.JPG') or img_item.endswith('.PNG')

    if img_item_path.is_file() and img_item_endswith:

        img_file_name = img_item_path.name
        img_item_data = {'fileNameBefore': img_file_name}
        img: Image.Image = Image.open(img_item_path)

        (shotname, extension) = os.path.splitext(img_file_name)

        byteSizeBefore = len(img.fp.read())
        img_item_data['byteSizeBefore'] = byteSizeBefore


        if byteSizeBefore < 307200:
            return


        if img_item.endswith('.png') or img_item.endswith('.PNG'):
            if jpga > 0:
                img = img.convert('RGB')
                extension = ".jpg"
            else:
                img = img.quantize(colors=256)

        save_file = "{}/{}{}".format(output, shotname, extension)
        img_item_data['fileNameAfter'] = save_file
        img.save(save_file, quality=quality, optimize=True, subsampling=subsampling)
        byteSizeAfter = os.path.getsize(save_file)
        img_item_data['byteSizeAfter'] = byteSizeAfter
        print("-" * 70)

    else:
        notfound_imgs.append(img_item)