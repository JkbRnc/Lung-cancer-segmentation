import os
import cv2
import util
from glob import glob
from albumentations import HorizontalFlip, VerticalFlip, Rotate


def load_data(path):
    x_train = sorted(glob(os.path.join(path, "train", "frames", "*.png")))
    y_train = sorted(glob(os.path.join(path, "train", "masks", "*.png")))

    x_test = sorted(glob(os.path.join(path, "test", "frames", "*.png")))
    y_test = sorted(glob(os.path.join(path, "test", "masks", "*.png")))

    return x_train, y_train, x_test, y_test


def augment_data(images, masks, save_path, augment=True):
    for data, target in zip(images, masks):
        name = data.split('/')[-1].split('.')[0]

        data = cv2.imread(data, cv2.IMREAD_COLOR)
        target = cv2.imread(target, cv2.IMREAD_GRAYSCALE)

        all_images = [data]
        all_masks = [target]

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=data, mask=target)
            all_images.append(augmented["image"])
            all_masks.append(augmented["mask"])

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=data, mask=target)
            all_images.append(augmented["image"])
            all_masks.append(augmented["mask"])

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=data, mask=target)
            all_images.append(augmented["image"])
            all_masks.append(augmented["mask"])

        for j, (img, msk) in enumerate(zip(all_images, all_masks)):
            img_name = f"{name}_{j}.png"
            image_path = os.path.join(save_path, "frames", img_name)
            mask_path = os.path.join(save_path, "masks", img_name)

            cv2.imwrite(image_path, img)
            cv2.imwrite(mask_path, msk)


def run_augmentation(data_path='./data/'):
    util.seeding(69)

    x_train, y_train, x_test, y_test = load_data(data_path)

    util.create_dir(util.TRAIN_PATH + "frames/")
    util.create_dir(util.TRAIN_PATH + "masks/")
    util.create_dir(util.TEST_PATH + "frames/")
    util.create_dir(util.TEST_PATH + "masks/")

    augment_data(x_train, y_train, util.TRAIN_PATH, augment=True)
    augment_data(x_test, y_test, util.TEST_PATH, augment=False)


if __name__ == "__main__":
    run_augmentation()
