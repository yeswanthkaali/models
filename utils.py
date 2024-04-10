from torch_lr_finder import LRFinder
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def lr_finder(train_loader,optimizer,criterion,model,end_lr):
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device="mps")
    lr_finder.range_test(train_loader,end_lr=end_lr, num_iter=100, step_mode="exp")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
from tqdm import tqdm
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
def train(model, device, train_loader, optimizer,scheduler,criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        correct += GetCorrectPredCount(output, target)
        processed += len(data)
        train_loss+=loss.item()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_accuracy=100*correct/processed
    train_loss=train_loss/len(train_loader)
    return train_loss,train_accuracy


def test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy=100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,test_accuracy

def plt_wrongpred(model,test_data,device,class_names,target_layers):
    model_eval=model.eval()
    wrong_predictions = []
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target
            output = model(data.unsqueeze(0))
            pred = torch.argmax(output, 1)# get the index of the max log-probability
            if(pred!=target):
                wrong_predictions.append((data, pred, target))
    device = "cpu"
    for i in range(1,min(11, len(wrong_predictions))):
        plt.subplot(2,5,i)
        plt.axis('off')
        image, predicted, label = wrong_predictions[i]
        image_ch = image.to(device).permute(1, 2, 0)  # Rearrange dimensions for plotting (assuming channels are last)    
        # Plot the image
        gm=gradcam(image,image,model,target_layers)
        plt.imshow(gm.clamp(0,1))
        plt.title(f"{class_names[predicted]}-{class_names[label]}")
def gradcam(image,input_tensor,model,target_layers):
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    return visualization

