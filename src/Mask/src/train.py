import torch
import torchvision

from dataset import WildParkMaskDataset, get_transform
from model import get_model_instance_segmentation
import utils
from engine import train_one_epoch, evaluate


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device.')

split = 'test'

img_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/frames/'
ann_path = '/home/smuuts/Documents/uni/PG/CenterTrack/data/Mask R-CNN/annotations/'

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = WildParkMaskDataset(img_path, ann_path, 'test', get_transform(train=True))
    dataset_test = WildParkMaskDataset(img_path, ann_path, 'test', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, pretrained=True)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

    print('Saving model...')
    torch.save(model.state_dict(), '/home/smuuts/Documents/uni/PG/CenterTrack/src/Mask/models/test.pth')

    print('Done!')


if __name__ == '__main__':
    main()