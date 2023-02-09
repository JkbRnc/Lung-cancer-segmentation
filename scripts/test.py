from glob import glob
import cv2
import numpy as np
import torch

import util
from unet import UNet
from tqdm import tqdm


def read(x, device, mask=False):
    if mask:
        x = np.expand_dims(x, axis=0)
    else:
        x = np.transpose(x, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch size (1)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    return x


def flatten(target):
    target = target.cpu().numpy()
    target = target > 0.5
    target = target.astype(np.uint8)
    return target.reshape(-1)


def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def run_tests(test_path_frames=util.VALID_PATH_FRAMES, test_path_masks=util.VALID_PATH_MASKS, checkpoint_path="files/checkpoint.pth"):
    util.seeding(69)
    util.create_dir("results")

    """ Load dataset """
    x_test = sorted(glob(test_path_frames))
    y_test = sorted(glob(test_path_masks)
)
    """ Hyperparameters """
    height = 512
    width = 512
    size = (height, width)

    """ Build model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for _, (data, target) in tqdm(enumerate(zip(x_test, y_test)), total=(len(x_test))):
        name = data.split("/")[-1].split(".")[0]

        """ Read image and mask """
        image = cv2.imread(data, cv2.IMREAD_COLOR)
        data = read(image, device)
        mask = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        target = read(mask, device, mask=True)

        with torch.no_grad():
            pred = model(data)
            pred = torch.sigmoid(pred)

            pred = pred[0].cpu().numpy()
            pred = np.squeeze(pred, axis=0)
            pred = pred > 0.5
            pred = np.array(pred, dtype=np.uint8)

        """ Save images """
        original_mask = parse_mask(mask)
        generated_mask = parse_mask(pred)
        line = np.ones(([size[1], 10, 3])) * 10

        cat_images = np.concatenate(
            [image, line, original_mask, line, generated_mask * 255], axis=1)
        cv2.imwrite(f"results/{name}.png", cat_images)

if __name__ == "__main__":
    run_tests()
