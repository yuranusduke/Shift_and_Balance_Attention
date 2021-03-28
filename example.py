# -*- coding: utf-8 -*-
"""
One Cifar10 example of the results
I use modified VGG16(less parameters in fc and with bn) and CIFAR-10
But I use SB at the end of each block unlike in original paper

Created by Kunhong Yu
Date: 2021/03/28
"""
import torch as t
import torchvision as tv
from SBAttention import weights_init
import tqdm
import fire
import matplotlib.pyplot as plt
from utils import VGG16
import datetime

############Hyper-parameters############
device = 'cuda' # learning device
attn_ratio = 0.5 # hidden size ratio
use_sb = True # False to enable SB attention
use_se = False # True to enable SE attention
activation = 'tanh' # tanh as default in SB attention

epochs = 100 # training epochs
batch_size = 32 # batch size
init_lr = 0.1 # initial learnign rate
gamma = 0.2 # learning rate decay, here we use step decay strategy for simplicity
milestones = [20, 40, 60, 80] # learning rate decay epochs
weight_decay = 1e-5 # weight decay
########################################

# Step 0 Decide the structure of the model#
# Step 1 Load the data set#
def main(**kwargs):
    """Simply run in one cell"""

    ######################
    #  Unfold parameters #
    ######################
    device = kwargs['device']
    attn_ratio = kwargs['attn_ratio']
    use_sb = kwargs['use_sb']
    use_se = kwargs['use_se']
    activation = kwargs['activation']

    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    init_lr = kwargs['init_lr']
    gamma = kwargs['gamma']
    milestones = kwargs['milestones']
    weight_decay = kwargs['weight_decay']

    device = t.device(device)

    transform = \
        tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding = 4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomRotation(15),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(0.5, 0.5)])

    train_data = tv.datasets.CIFAR10(root = './data',
                                     download = False,
                                     train = True,
                                     transform = transform)
    test_data = tv.datasets.CIFAR10(root = './data',
                                    download = False,
                                    train = False,
                                    transform = tv.transforms.Compose([
                                        tv.transforms.ToTensor(),
                                        tv.transforms.Normalize(0.5, 0.5)
                                    ]))

    train_loader = t.utils.data.DataLoader(train_data,
                                           shuffle = True,
                                           batch_size = batch_size)
    test_loader = t.utils.data.DataLoader(test_data,
                                          shuffle = False,
                                          batch_size = batch_size)

    # Step 2 Reshape the inputs#
    # Step 3 Normalize the inputs#
    # Step 4 Initialize parameters#
    # Step 5 Forward propagation(Vectorization/Activation functions)#

    global model
    model = VGG16(use_sb = use_sb, device = device, attn_ratio = attn_ratio, use_se = use_se, activation = activation)
    model.apply(weights_init)
    model.to(device)
    print('VGG model : \n', model)

    # Step 6 Compute cost#
    loss = t.nn.CrossEntropyLoss().to(device)
    # Step 7 Backward propagation(Vectorization/Activation functions gradients)#
    optimizer = t.optim.SGD(filter(lambda x : x.requires_grad, model.parameters()),
                            lr = init_lr, momentum = 0.9, weight_decay = weight_decay, nesterov = True)

    lr_scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    gamma = gamma,
                                                    milestones = milestones)

    def eval(model, eval_iter, device, loss = None):
        """This function is used to evaluate model
        Args :
            --model: model instance
            --eval_iter: evaluation data iter
            --device
            --loss: default is None
        return :
            --eval_loss: eval loss
            --eval_acc: eval acc
        """
        model.eval()
        with t.no_grad():
            count = 0.
            eval_loss = 0.
            eval_acc = 0.
            for i, (batch_x, batch_y) in enumerate(eval_iter):
                batch_x = batch_x.view(batch_x.size(0), 3, 32, 32)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                out = model(batch_x)
                preds = t.argmax(out, dim = 1)
                if loss is not None:
                    batch_loss = loss(out, batch_y)
                    eval_loss += batch_loss.item()

                correct = t.sum(batch_y == preds).float()
                batch_acc = correct / batch_x.size(0)
                eval_acc += batch_acc.item()

                count += 1.

            eval_acc /= count
            if loss is not None:
                eval_loss /= count

                return eval_loss, eval_acc

            return eval_acc

    # Step 8 Update parameters#
    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    print('\nStart training...')
    for epoch in tqdm.tqdm(range(epochs)):
        print('Epoch %d / %d.' % (epoch + 1, epochs))
        print('Current learning rate : ', optimizer.state_dict()['param_groups'][0]['lr'])
        epoch_loss = 0.
        epoch_acc = 0.
        count = 0.
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.view(batch_x.size(0), 3, 32, 32)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            batch_loss = loss(out, batch_y)
            batch_loss.backward()
            optimizer.step()

            if i % batch_size == 0:
                count += 1.
                preds = t.argmax(out, dim = 1)
                correct = t.sum(batch_y == preds).float()
                acc = correct / batch_x.size(0)
                print('\t\033[4;33m Training Batch INFO :\033[0m Batch %d has loss : %.3f --> acc : %.2f%%.' % (
                    i + 1, batch_loss.item(), acc.item() * 100.
                ))

                epoch_acc += acc.item()
                epoch_loss += batch_loss.item()

        epoch_loss /= count
        epoch_acc /= count
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        eval_loss, eval_acc = eval(model,
                                   eval_iter = test_loader,
                                   device = device, loss = loss)
        model.train()
        eval_losses.append(eval_loss)
        eval_accs.append(eval_acc)
        print('\033[31m Training Epoch INFO :\033[0m This epoch has train loss : %.3f --> train acc : %.2f%% || '
              'eval loss : %.3f --> eval acc : %.2f%%.' % (
            epoch_loss, epoch_acc * 100., eval_loss, eval_acc * 100.
        ))

        lr_scheduler.step()


    print('Training is done!\n')
    # Step 9 Make a test#
    final_test_acc = eval(model, test_loader, device)
    print('Final test acc : {:.2f}%.'.format(final_test_acc * 100.))
    with open('./results/results.txt', 'a+') as f:
        model_string = str(datetime.datetime.now()) + ' ::: '
        model_string += '[attn_ratio : ' + str(attn_ratio) + ', use_sb : ' + str(use_sb) + ', use_se : ' + str(use_se) + ', activation : ' + activation + '] --> acc : %.2f%%.' % (final_test_acc * 100.) + '\n'
        f.write(model_string)

    t.save(model, './results/vgg16_' + str(attn_ratio) + '_' + str(use_sb) + '_' + str(use_se) + '_' + activation + '.pth')

    # visualize
    f, ax = plt.subplots(1, 2, figsize = (20, 6))
    f.suptitle('Training and eval statistics')

    ax[0].plot(range(len(train_losses)), train_losses, label = 'training_loss')
    ax[0].plot(range(len(eval_losses)), eval_losses, label = 'eval_loss')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Value')
    ax[0].set_title('Losses')
    ax[0].grid(True)
    ax[0].legend(loc = 'best')

    ax[1].plot(range(len(train_accs)), train_accs, label = 'training_acc')
    ax[1].plot(range(len(eval_accs)), eval_accs, label = 'eval_acc')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Value')
    ax[1].set_title('Accs')
    ax[1].grid(True)
    ax[1].legend(loc = 'best')

    plt.savefig('./results/vgg16_' + str(attn_ratio) + '_' + str(use_sb) + '_' + str(use_se) + '_' + activation + '.png')
    plt.close()




if __name__ == '__main__':
    fire.Fire()

    """
    Usage:
    python example.py main --device='cuda' --attn_ratio=0.5 --use_sb=True --use_se=True --activation='tanh' --epochs=100 --batch_size=20 --init_lr=0.1 --gamma=0.2 --milestones=[20,40,60,80] --weight_decay=1e-5 
    you can also import SB module like:
    from SBAttention import SBAttention as sbattn
    """


    print('\nDone!\n')