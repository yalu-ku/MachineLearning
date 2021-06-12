"""
Transfer learning
with chest_xray DATA
Implemented by ARO
"""

from __future__ import print_function, division

import os, time, copy, cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

""" 1. Load Data """
# 데이터 셋 경로 지정
root_path = os.getcwd()  # 현재 위치 확인
data_path = root_path + '/chest_xray'
train_path = data_path + '/train'
val_path = data_path + '/val'
test_path = data_path + '/test'

# 클래스 정보 지정
class_names = os.listdir(train_path)
class_num = len(class_names)

# train / val / test 폴더의 각 이미지 개수 확인
for class_name in class_names:
    train_sub_path = train_path + '/' + class_name
    val_sub_path = val_path + '/' + class_name
    test_sub_path = test_path + '/' + class_name
    print('The number of ' + class_name + ' dataset(Train) : ', len(os.listdir(train_sub_path)))
    print('The number of ' + class_name + ' dataset(Validation) : ', len(os.listdir(val_sub_path)))
    print('The number of ' + class_name + ' dataset(Test) : ', len(os.listdir(test_sub_path)))
"""
The number of NORMAL dataset(Train) :  1341
The number of NORMAL dataset(Validation) :  8
The number of NORMAL dataset(Test) :  234
The number of PNEUMONIA dataset(Train) :  3875
The number of PNEUMONIA dataset(Validation) :  8
The number of PNEUMONIA dataset(Test) :  390
"""

# 이미지 확인해보기
image = cv2.imread(val_sub_path + '/person1946_bacteria_4874.jpeg')
plt.show()

# 이미지 정보 확인
type(image)
print(image.shape)


# Data Augmentation
data_transform = {  # image transformation (dictionary)
    'train': transforms.Compose([
        transforms.Resize(299),
        # transforms.CenterCrop(299),
        # transforms.RandomResizedCrop(224)  # 이미지를 랜덤하게 사이즈 변경 [(가로, 세로) : (224, 224)] cropping(자르기)
        #     transforms.RandomHorizontalFlip(),  # 가로축 방향으로 flipping(뒤집기)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),  # 이미지 크기 resize
        transforms.CenterCrop(299),  # 가운데를 기준으로 cropping
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB에 대한 평균값과 standard diviation
    ])
    # 'test': transforms.Compose([
    #     transforms.RandomResizedCrop(224)
    # ])
}

# data_dir = './chest_xray/'
# print(os.path.join(data_dir), 'train')

# ImageFolder를 이용하여 구조화된 디렉터리를 만들어서 data set에 넣고 사용
image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transform[x])
                  for x in ['train', 'val']}
# print(image_datasets)
print('The number of dataset(Train) after Augmentation : ', len(image_datasets['train']))
print('The number of dataset(Validation) after Augmentation : ', len(image_datasets['val']))

# data loader로 불러오기
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Set device (cuda Env.)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Training the model """

# scheduler,
'''
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    # 모델의 weight, bias 등 모든 정보를 Copy해옴
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # initialization

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train과 validation을 한번씩(overfitting 모니터링하기에 좋음)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()  # train과 validation할때 모드를 바꾸어줌

            running_loss = 0.0
            running_corrects = 0

            # data load에서 mini-batch만큼 가져와서 GPU로 보냄
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Initialize optimizer

                # train mode일때 gradient를 enable (back-prop을 하겠다는 뜻)
                # 즉, train mode 일때만 back-prop 진행.
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # outputs = model(inputs)  # forward-prop (train/val 모두)
                    # # preds = torch.max(outputs)
                    # # _, preds = torch.max(outputs, 1)  # 1축 방향으로 index를 가져옴 -> acc 정할때 사용
                    # loss = criterion(outputs, labels)
                    #

                    # train 일때만 backward하고 update 시킴
                    # backward-prop + optimize (only train set)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()  # learning decay (train일때만 필요)

                    # loss값과 accuracy 계산
                    # statistics(?)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # if phase == 'train':
                #     scheduler.step()  # learning decay (train일때만 필요)

                # mini-batch에 대한 loss값
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # best model 찾기
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since  # 실행 시간
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model, val_acc_history  # best model을 return

    # image와 label을 함께 보여주는 함수

#
# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 plt.imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
#
#
# """ Transfer learning """
#
#
# def main():
#     model_ft = model_ft.to(device)
#     params_to_update = model_ft.parameters()
#     if feature_extract:
#         params_to_update = []
#         for name, param in model_ft.named_parameters():
#             params_to_update.append(param)
#             print('\t', name)
#         else:
#             for name, param in model_ft.named_parameters():
#                 if param.requires_grad == True:
#                     print('\t', name)
#
#     # model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
#     model_ft = models.inception_v3(pretrained=True)
#     set_parameter_requires_grad(model_ft, feature_extract=True)
#     num_ftrs = model_ft.fc.in_features
#     print(num_ftrs)  # -개의 fully connected layer
#     print(model_ft.fc.out_features)  # 원래 이렇게 생김
#
#     # 마지막 fully connected layer를 원하는대로 바꾸어줌
#     # 현재 데이터에서는 2개의 클래스만 구별하면 됨
#     # model_ft.fc = nn.Linear(num_ftrs, 2)
#     model_ft.AuxLogits.fc = nn.Linear(768, num_ftrs)
#     model_ft.fc = nn.Linear(512, num_ftrs)
#
#
#     model_ft = model_ft.to(device)
#
#     criterion = nn.CrossEntropyLoss()
#
#     # 마지막 바꾼 레이어를 다시 optimization
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
#     # epoch 7번 마다 감마를 0.1로 줄여가겠다고 정의
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
#     """ Train function CALL """
#     # model_ft 는 best model 을 return 함
#     # exp_lr_scheduler,
#     model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft,  num_epochs=25, is_inception=(model_ft=="inception"))
#
#
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
# """ Graph """
#
# if __name__ == '__main__':
#     main()
'''