
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    size = len(data_loader)
    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch % 10 == 0:
            log(epoch, loss_dict.copy(), size, batch)
        

def log(epoch, losses, batch_amt, batch_current):
    losses = {key:losses[key].detach().to('cpu').item() for key in losses}

    done = int(batch_current/batch_amt * 100)
    message = [f'Epoch {epoch} [{batch_current}/{batch_amt}] {done}%,  loss: ']
    for key in losses:
        val = f' {key}: {losses[key]:.4f} '
        message.append(val)
    print(''.join(message))
