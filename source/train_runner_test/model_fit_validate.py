import config
import torch

def fit(model, dataloader, data, optimizer, criterion):
    # print("training")
    model.train()
    train_running_loss = 0.0
    counter = 0
    num_batches = int(len(data) / dataloader.batch_size)
    for i, data in enumerate(dataloader):
        counter += 1
        image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
            config.DEVICE
        )
        keypoints = keypoints.view(keypoints.size(0), -1)
        # print(keypoints)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


def validate(model, dataloader, data, criterion):
    # print("validate")
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    num_batches = int(len(data) / dataloader.batch_size)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            counter += 1
            image, keypoints = data["image"].to(config.DEVICE), data["keypoints"].to(
                config.DEVICE
            )
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()
            # if (epoch + 1) % 25 == 0 and i == 0:
    valid_loss = valid_running_loss / counter
    return valid_loss
