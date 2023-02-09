import time
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util
from unet import UNet
from loss import DiceBCELoss, DiceLoss
from data_loader import BjorkeDataset


def train(model, loader, optimizer, scaler, loss, device):
    epoch_loss = 0.0
    model.train()

    for data, target in loader:
        data = data.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        """ Forward """
        with torch.cuda.amp.autocast():
            preds = model(data)
            current_loss = loss(preds, target)

        """ Backward """
        optimizer.zero_grad()
        scaler.scale(current_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        """ Update loss """
        epoch_loss = epoch_loss + current_loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss, device):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            preds = model(data)
            epoch_loss = epoch_loss + loss(preds, target).item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def run_train(train_path_frames=util.TRAIN_PATH_FRAMES, train_path_masks=util.TRAIN_PATH_MASKS, valid_path_frames=util.VALID_PATH_FRAMES, valid_path_masks=util.VALID_PATH_MASKS, batch_size=2, num_epochs=50, lr=1e-4, num_workers=2, checkpoint_path="files/checkpoint.pth"):
    """ Seeding """
    util.seeding(69)

    """ Dirs """
    util.create_dir("files")

    """ Load dataset """
    x_train = sorted(glob(train_path_frames))
    y_train = sorted(glob(train_path_masks))

    x_valid = sorted(glob(valid_path_frames))
    y_valid = sorted(glob(valid_path_masks))

    """ Loader """
    train_dataset = BjorkeDataset(x_train, y_train)
    valid_dataset = BjorkeDataset(x_valid, y_valid)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    """ Build model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = DiceBCELoss()
    # loss = nn.BCEWithLogitsLoss()
    # loss = DiceLoss()
    scaler = torch.cuda.amp.GradScaler()

    """ Train model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer,
                           scaler, loss, device)
        valid_loss = evaluate(model, valid_loader, loss, device)

        """ Save model """
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        mins, secs = util.epoch_time(start_time=start_time, end_time=end_time)
        print(
            f'Epoch: {epoch + 1}, elapsed time: {mins}:{secs}\nTrain loss: {train_loss}\nValid loss: {valid_loss}')


if __name__ == "__main__":
    run_train()
