import argparse
import json
from math import ceil

import PIL
import numpy as np
import torch
from torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--image', dest="image", type=str, required=True)
    parser.add_argument('--checkpoint', dest="checkpoint", type=str, required=True)
    parser.add_argument('--top_k', dest="topk", type=int, default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

    args = parser.parse_args()

    return args


def process_image(image):
    img = PIL.Image.open(image)

    original_width, original_height = img.size

    if original_width < original_height:
        size = [256, 256 ** 600]
    else:
        size = [256 ** 600, 256]

    img.thumbnail(size)

    center = original_width / 4, original_height / 4
    left, top, right, bottom = center[0] - (244 / 2), center[1] - (244 / 2), center[0] + (244 / 2), center[1] + (
            244 / 2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img) / 255

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std

    numpy_img = numpy_img.transpose(2, 0, 1)

    return numpy_img


def print_probability(probs, flowers):
    for i, k in enumerate(zip(flowers, probs)):
        print("Rank {}:".format(i + 1),
              "Flower: {}, Probability: {}%".format(k[1], ceil(k[0] * 100)))


def predict(image, model, category_names, top_k=5):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model.to("cpu")

    model.eval()

    torch_image = torch.from_numpy(np.expand_dims(process_image(image), axis=0)).type(torch.FloatTensor).to("cpu")

    logarithmic_probability = model.forward(torch_image)
    linear_probability = torch.exp(logarithmic_probability)
    top_probability, top_labels = linear_probability.topk(top_k)
    top_probability = np.array(top_probability.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probability, top_labels, top_flowers


def main():
    args = arg_parser()

    checkpoint = torch.load(args.checkpoint)

    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    image_tensor = process_image(args.image)

    top_probs, top_labels, top_flowers = predict(image_tensor, model, args.category_names, args.top_k)

    print_probability(top_flowers, top_probs)


if __name__ == '__main__':
    main()
