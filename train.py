import argparse
import json
from collections import OrderedDict
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--save', dest="save", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=18)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)

    train_Loader_size = len(train_loader)
    valid_Loader_size = len(validation_loader)
    test_loader_size = len(test_loader)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(25088, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('hidden_layer1', nn.Linear(args.hidden_units, args.hidden_units - 30)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(args.hidden_units - 30, args.hidden_units - 50)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(args.hidden_units - 50, train_Loader_size - 1)),
        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier

    if torch.cuda.is_available() and args.gpu:
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            print("Batch no: {:03d}, Loss on training: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

    with torch.no_grad():

        model.eval()

        for j, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            valid_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            valid_acc += acc.item() * inputs.size(0)

            print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(),
                                                                                                       acc.item()))

    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save({'structure': 'alexnet',
                'hidden_layer1': args.hidden_units,
                'dropout': 0.5,
                'epochs': args.epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer_dict': optimizer.state_dict()},
               args.save)


if __name__ == "__main__":
    main()
