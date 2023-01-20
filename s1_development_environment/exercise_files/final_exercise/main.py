import argparse
import sys
import os
import torch
import click
from tqdm import trange

from data import MnistDataset
from model import MyNetwork


@click.group()
def cli():
    pass


@click.command()
@click.argument("datapath")
@click.argument("checkpoint_path")
@click.option("--input_size", default=784, help='input size of data')
@click.option("--output_size", default=10, help='number of output classes')
# @click.option("--hidden_layers", default=[256, 128, 64], help='number of units per hidden layer')

@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=30, help='number of epochs for training')

def train(datapath, checkpoint_path, input_size, output_size, lr, epochs):
    print("Training day and night")
    print(lr)

    # Get model
    model = MyNetwork(input_size, output_size, [256, 128, 64], drop_p=0.5)
    # Get data
    train_set = MnistDataset(dataset_dir=datapath, train=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = MnistDataset(dataset_dir=datapath, train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    # Define Loss function & Optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    curr_best_loss = 1000

    with trange(epochs, unit="carrots") as pbar:
        for e in pbar:
        # for e in range(epochs):

            pbar.set_description(f"Epoch {e}")
            ## Training Loop
            running_train_loss = 0
            # set model to train mode
            model.train()

            for images, labels in trainloader:
                optimizer.zero_grad()
                
                # Flatten images into a 784 long vector
                images.resize_(images.size()[0], 784)

                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item()
            else:
                mean_train_loss = running_train_loss/len(trainloader)
                print(f'\nTrain Loss: {mean_train_loss}')
                train_losses.append(mean_train_loss)
            
            ## Validation Loop
            mean_loss, _ = validation(model, testloader, criterion)
            test_losses.append(mean_loss)
            pbar.set_postfix(train_loss=mean_train_loss, valid_loss=mean_loss)

            # If result is better than current best -> Save model
            if mean_loss < curr_best_loss:
                checkpoint = {'input_size': 784,
                                'output_size': 10,
                                'hidden_layers': [each.out_features for each in model.hidden_layers],
                                'state_dict': model.state_dict()}

                torch.save(checkpoint, os.path.join(checkpoint_path, 'checkpoint.pth'))



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyNetwork(checkpoint['input_size'],
                      checkpoint['output_size'],
                      checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def validation(model, testloader, criterion=None):
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()
        for images, labels in testloader:

            images = images.resize_(images.size()[0], 784)

            output = model.forward(images)
            if criterion:
                test_loss += criterion(output, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Get the most likely class
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    mean_acc = accuracy/len(testloader)
    mean_loss = test_loss/len(testloader)
    print(f'Mean Test Loss: {mean_loss}')
    print(f'Mean Test Accuracy: {mean_acc*100}%\n')

    return mean_loss, mean_acc

@click.command()
@click.argument("model_checkpoint")
@click.argument("datapath")
def evaluate(model_checkpoint, datapath):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    # Get model
    model = load_checkpoint(model_checkpoint)
    # Get test data    
    test_set = MnistDataset(dataset_dir=datapath, train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    mean_loss, mean_accuracy = validation(model, testloader)



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    