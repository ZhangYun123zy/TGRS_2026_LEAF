from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from model import EHSnet
import numpy as np
from utils_HSI import sample_gt,sample_gt_fixed, metrics, get_device, seed_worker
from datasets import HyperX, get_dataset
import os
import torch.utils.data as data
import scipy.io as io
from sklearn.metrics import classification_report
import clip
import time
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import json
parser = argparse.ArgumentParser(description='PyTorch LEAF')

parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='/mnt/HDD/data/ZY/EHSnet-net/TGRS_2026_LEAF/datasets/',
                    help='the path to load the data')
parser.add_argument('--dataset', type=str, default='Houston',
                    help='the name of dataset')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--num_epoch', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')

# Training option
group_train = parser.add_argument_group('Training')
group_train.add_argument('--a', type=float, default=0,
                         help="Regularization parameter, controlling the contribution of different type of classsification "
                              "features.")
group_train.add_argument('--alpha', type=float, default=0.2,
                         help="Regularization parameter, controlling the contribution of different type of text "
                              "features.")
group_train.add_argument('--alpha2', type=float, default=0.8,
                         help="Regularization parameter, controlling the contribution of different type of text "
                              "features.")
group_train.add_argument('--beta', type=float, default=0,
                         help="Regularization parameter, controlling the contribution of loss_clip ")
group_train.add_argument('--gama', type=float, default=0,
                         help="Regularization parameter, controlling the contribution of loss_global ")
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=0.05,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int, default=64,
                         help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

parser.add_argument('--with_exploration', default=True, action='store_true',
                    help="See data exploration visualization")

parser.add_argument('--repeat_time', type=int, default=1, help='Number of matched base/incremental runs')
parser.add_argument('--gpt_path', type=str, default=None, help='JSON text descriptions')
args = parser.parse_args()
DEVICE = get_device(args.cuda)

datasets_list = {
    'Pavia': {'source_name': 'paviaU', 'target_name': 'paviaC', 'source_label_name': 'paviaU_7gt',
              'target_label_name': 'paviaC_7gt'},
    'Houston': {'source_name': 'Houston13', 'target_name': 'Houston18', 'source_label_name': 'Houston13_7gt',
                'target_label_name': 'Houston18_7gt'},
    'OHID': {'source_name': 'OHID1', 'target_name': 'OHID2', 'source_label_name': 'OHID1_gt',
                'target_label_name': 'OHID2_gt'},
}

description_list = {
    "Pavia": 'A hyperspectral image of Pavia University with the surface covered with trees, asphalt, brick, '
             'bitumen,shadow, meadow, and bare soil',
    "Houston": 'A hyperspectral image of the University of Houston with the surface covered with grass healthy, '
               'grass stressed, trees, water, residential buildings, non-residential buildings and road',
    "OHID": 'A hyperspectral image of the Zhuhai City captured by Orbita with the surface covered with building, '
               'farmland, road, water, fishpond',
}

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([140, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 200]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([216, 191, 216]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 255]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 200, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        
    return y

def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def get_cls_map(y_pred,y,correct,data):
    #y_pred, y_new = test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()



    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    classification_map(y_re, y, 300,
                       'classification_maps/' + '{}_predictions_{}.png'.format(data,correct))
    classification_map(gt_re, y, 300,
                       'classification_maps/' + '{}_gt.png'.format(data))
    print('------Get classification maps successful-------')

def get_all_map(model,y_all_gt,acc,test_loader,text_1, text_2,source_name):
    y_all_pred=[]
    y_all=[]
    model.eval()
    _, _, y_all_pred, _ = test(model,test_loader,text_1, text_2) 
    y_all_pred = np.concatenate(y_all_pred)
    #print(np.max(y_all_pred),np.min(y_all_pred),np.max(y_all),np.min(y_all),np.max(y_all_gt),np.min(y_all_gt))
    get_cls_map(y_all_pred,y_all_gt,acc,source_name)

def train(epoch, model,train_loader, num_epoch, text_1, text_2):
    '''
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if (epoch - 1) % 10 == 0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True,
                                    weight_decay=0.0005)
    correct = 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader

    text_1 = text_1.to(DEVICE)
    text_2 = text_2.to(DEVICE)
    for i in range(1, num_iter):

        model.train()
        data_src, label_src = iter_source.__next__()
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        label_src = label_src - 1

        optimizer.zero_grad()


        loss_clip,loss_clip_proto, label_src_pred,label_src_enhance_pred = model(data_src, text_1, text_2, label_src, text_ratio=args.alpha)
        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
        loss_ehanced_cls = F.nll_loss(F.log_softmax(label_src_enhance_pred, dim=1), label_src.long())
        loss = loss_cls + args.a*loss_ehanced_cls+ args.beta * loss_clip + args.gama * loss_clip_proto

        loss.backward()
        optimizer.step()

        pred = label_src_pred.data.max(1)[1]
        correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch, i * len(data_src), len_src_dataset,
                                                             100. * i / len_src_loader))
            print('loss: {:.6f},  loss_cls: {:.6f},  loss_clip: {:.6f},  loss_clip: {:.6f}'.format(
                loss.item(), loss_cls.item(), loss_clip.item(), loss_clip_proto.item()))
    acc = correct.item() / len_src_dataset
    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, acc, len_src_dataset))

    return model


def test(model,test_loader,text_1, text_2):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    len_tar_dataset = len(test_loader.dataset)
    text_1 = text_1.to(DEVICE)
    text_2 = text_2.to(DEVICE)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            loss_temp,_, label_src_pred,_ = model(data, text_1, text_2, label)
            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long()).item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        print(
            'Average test loss: {:.4f},test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
                loss, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))

    return correct, correct.item() / len_tar_dataset, pred_list, label_list

def my_cal_oa_aa_ka(true,pred):
    Kappa = cohen_kappa_score(np.array(true).reshape(-1, 1), np.array(pred).reshape(-1, 1))
    OA = (pred==true).mean()
    num_label = np.unique(true)
    N = len(num_label)
    avec = []
    for i in range(N):
        class_idx = np.where(true==num_label[i])
        class_pred = pred[class_idx]
        class_true = true[class_idx]
        class_acc = (class_true==class_pred).mean()
        avec.append(class_acc)
    mean_class = np.mean(np.asarray(avec))
    #print('aa: ',mean_class)
    OA = np.round(np.array(OA)* 100,2)
    mean_class= np.round(np.array(mean_class)* 100,2)
    Kappa = np.round(np.array(Kappa)* 100,2)
    avec = np.round(np.asarray(avec) * 100, 2)
    return OA,mean_class,Kappa,avec

if __name__ == '__main__':
    args.save_path = args.save_path + args.dataset + f'/alpha{args.alpha}_beta{args.alpha2}_e{args.num_epoch}'
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    args.data_path = args.data_path + args.dataset + '/'
    print('data_path:', args.data_path)
    source_name = datasets_list[args.dataset]['source_name']
    target_name = datasets_list[args.dataset]['target_name']
    seeds=[1337, 1220, 1336, 1330, 1224, 1236, 1226, 1235, 1233, 1229]
    repeat_time=args.repeat_time
    if args.dataset=="Houston":
        record_results=np.zeros(((7+3)*2+1,repeat_time), dtype=object)
        print_matrix = np.zeros(((7+3)*2+1, repeat_time + 1), dtype=object)
    elif args.dataset=="Pavia":
        record_results=np.zeros(((7+3)*2+1,repeat_time), dtype=object)
        print_matrix = np.zeros(((7+3)*2+1, repeat_time + 1), dtype=object)
    elif args.dataset=="OHID":
        record_results=np.zeros(((5+3)*2+1,repeat_time), dtype=object)
        print_matrix = np.zeros(((5+3)*2+1, repeat_time + 1), dtype=object)

    gpt_path=args.gpt_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.dataset}.json")
    file = open(gpt_path, "r")
    GPT_prompt_dict=json.load(file)
    finegraind_descriptions = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}

    if not 1 <= repeat_time <= len(seeds):
        raise ValueError(f"repeat_time must be 1..{len(seeds)}")
    for repeat in range(repeat_time):
        seed=seeds[repeat]
        seed_worker(seed)
        print('load source dataset:', end=' ')
        img_src, gt_src, label_names_src, ignored_labels = get_dataset(source_name, args.data_path)
        print('load target dataset:', end=' ')
        img_tar, gt_tar, label_names_tar, ignored_labels = get_dataset(target_name, args.data_path)


        text_1 = torch.cat([clip.tokenize(f'A hyperspectral image of {label_names_src[k]}') for k in range(len(label_names_src))])
        
        all_prompts = []
        for classname in label_names_src:
            single_class_prompts = finegraind_descriptions[classname.lower().replace("_", " ")]
            all_prompts.extend(single_class_prompts)

        text_2 = torch.cat([clip.tokenize(p) for p in all_prompts])

        all_gt_src=gt_src
        all_gt_tar=gt_tar

        num_classes = gt_src.max()
        N_BANDS = img_src.shape[-1]
        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': ignored_labels,
                            'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        r = int(hyperparams['patch_size'] / 2) + 1
        img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
        img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
        gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
        gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

        train_gt_src, test_gt_src_con, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        #test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
        train_gt_tar, test_gt_tar, _, _ = sample_gt_fixed(gt_tar, 5, mode='random')
        img_src_con, train_gt_src_con = img_src, train_gt_src

        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

        hyperparams_train = hyperparams.copy()
        hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

        train_dataset_src = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
        test_dataset_src= HyperX(img_src, test_gt_src_con, **hyperparams)
        train_dataset_tar = HyperX(img_tar, train_gt_tar, **hyperparams)
        test_dataset_tar = HyperX(img_tar, test_gt_tar, **hyperparams)
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader_src = data.DataLoader(train_dataset_src,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)
        test_loader_src= data.DataLoader(test_dataset_src,
                                    pin_memory=True,
                                    batch_size=64)
        
        train_loader_tar = data.DataLoader(train_dataset_tar,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)  
        test_loader_tar = data.DataLoader(test_dataset_tar,
                                    pin_memory=True,
                                    batch_size=64)

        src_all_dataset = HyperX(img_src, gt_src, **hyperparams)
        src_all_loader = torch.utils.data.DataLoader(src_all_dataset,
                                            pin_memory=True,
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=False)       

        tar_all_dataset = HyperX(img_tar, gt_tar, **hyperparams)
        tar_all_loader = torch.utils.data.DataLoader(tar_all_dataset,
                                            pin_memory=True,
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=False) 
        len_src_loader = len(train_loader_src)
        len_src_dataset = len(train_loader_src.dataset)

        len_tar_dataset = len(test_loader_tar.dataset)
        len_tar_loader = len(test_loader_tar)

        print(hyperparams)
        print("train samples :", len_src_dataset)
        print("target train samples :", len(train_loader_tar.dataset))
        print("test samples :", len_tar_dataset)

        correct, acc = 0, 0
        pretrained_dict = torch.load('./ViT-B-32.pt', map_location="cpu").state_dict()
        embed_dim = pretrained_dict["text_projection"].shape[1]
        context_length = pretrained_dict["positional_embedding"].shape[0]
        vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
        transformer_width = pretrained_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = 3
        model = EHSnet(embed_dim,
                    img_src.shape[-1], hyperparams['patch_size'], int(gt_src.max()),
                    context_length, vocab_size, transformer_width, transformer_heads,
                    transformer_layers).to(DEVICE)
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in pretrained_dict:
                del pretrained_dict[key]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        text_encoder_modules = [
                    model.token_embedding,
                    model.transformer,
                    model.ln_final
                ]
        
        for module in text_encoder_modules:
            for param in module.parameters():
                param.requires_grad_(False)
        
        model.positional_embedding.requires_grad_(False)
        model.text_projection.requires_grad_(False)
        model.logit_scale.requires_grad_(False)

        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')
        total_notrainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
        print(f'{total_notrainable_params / (1024 * 1024):.2f}M not training parameters.')

        description = description_list[args.dataset]
        print(f'global description of the {args.dataset} : \'{description}\'')

        train_start=time.time()
        for epoch in range(1, args.num_epoch + 1):
            t1 = time.time()
            model = train(epoch, model,train_loader_src, args.num_epoch, text_1, text_2)
            t2 = time.time()
            print('epoch time:', t2 - t1)
        train_end=time.time()
        train_time=train_end-train_start
        print('train time:', train_time)
        torch.save({'model':model.state_dict()}, os.path.join(args.save_path+f'/seed{seed}lr{args.lr}_a{args.a}_beta{args.beta}_gama{args.gama}_best.pkl'))