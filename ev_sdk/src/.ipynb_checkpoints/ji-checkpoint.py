import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


from model import BaseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    pth = '/usr/local/ev_sdk/model/best.pth'
    model = BaseModel(model_name='rep-a2')
    model.load_state_dict(torch.load(pth)['net'])
    model.to(device)
    model.eval()
    
    return model


class_dict = {0: 'airplane', 1: 'banana', 2: 'baseball', 3: 'bicycle', 4: 'bird', 5: 'book', 6: 'bulldozer', 7: 'cake', 8: 'camel', 9: 'camera', 10: 'cannon', 11: 'car', 12: 'cat', 13: 'chair', 14: 'computer', 15: 'cookie', 16: 'crown', 17: 'dog', 18: 'ear', 19: 'eye', 20: 'fish', 21: 'flower', 22: 'hand', 23: 'hat', 24: 'horse', 25: 'keyboard', 26: 'key', 27: 'knife', 28: 'ladder', 29: 'monkey', 30: 'mouse', 31: 'nose'}

size = 256
trans = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def process_image(net, input_image, args=None):
    img = input_image[..., ::-1]
    img = Image.fromarray(img)
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    out = net(img)
    _, pred = torch.max(out, 1)
    
    return json.dumps(
        {'class': class_dict[pred[0].item()]}
    )

#     return json.dumps({'class': 'airplane'})


if __name__ == '__main__':
    net = init()
    x = torch.randn((112, 112, 3)).numpy()
    print(process_image(net, x))