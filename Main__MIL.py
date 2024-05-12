import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os

from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import First_stream_Classifier, Second_stream_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric
import csv
import faiss
import pandas as pd
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='CBDS-MIL')
testMask_dir = '../../dataSets/reference.csv'

parser.add_argument('--name', default='CBDS-MIL', type=str)
parser.add_argument('--EPOCH', default=150, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr2', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='', type=str)
parser.add_argument('--mDATA_dir_val0', default='', type=str)
parser.add_argument('--mDATA_dir_test0', default='', type=str)
parser.add_argument('--TCGA_train', default='', type=str)
parser.add_argument('--TCGA_val', default='', type=str)
parser.add_argument('--TCGA_test', default='', type=str)
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=1, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='MIP', type=str)
parser.add_argument('--pooling_type', default='att', type=str)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def main():
    params = parser.parse_args()
    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))


    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    FiCls = First_stream_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate, device = params.device).to(params.device)


    if params.isPar:
        FiCls = torch.nn.DataParallel(FiCls)
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)

    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log.txt')
    save_dir = os.path.join(params.log_dir, 'best_model.pth')


    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')

    with open(params.mDATA0_dir_train0, 'rb') as f:
        mDATA_train = pickle.load(f)
    with open(params.mDATA_dir_val0, 'rb') as f:
        mDATA_val= pickle.load(f)
    with open(params.mDATA_dir_test0, 'rb') as f:
        mDATA_test = pickle.load(f)
    SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA(mDATA_train,params.numGroup)
    SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA(mDATA_val, params.numGroup)
    SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_test(mDATA_test,params.numGroup)

    # with open(params.TCGA_train, 'rb') as f:
    #     TCGA_train = pickle.load(f)
    # with open(params.TCGA_val, 'rb') as f:
    #     TCGA_val = pickle.load(f)
    # with open(params.TCGA_test, 'rb') as f:
    #     TCGA_test = pickle.load(f)
    # SlideNames_train, FeatList_train, Label_train = loder_dataSets(TCGA_train,numGroup=params.numGroup)
    # SlideNames_val, FeatList_val, Label_val = loder_dataSets(TCGA_val, numGroup=params.numGroup)
    # SlideNames_test, FeatList_test, Label_test = loder_dataSets(TCGA_test,numGroup=params.numGroup)

    print_log(f'training slides: {len(SlideNames_train)}, test slides: {len(SlideNames_test)}', log_file)

    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    if params.pooling_type == 'att':
        trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    optimizer_adam = torch.optim.Adam(FiCls.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(trainable_parameters, lr=params.lr2, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    best_auc = 0
    best_epoch = -1

    for ii in range(params.EPOCH):

        for param_group in optimizer_adam.param_groups:
            curLR = param_group['lr']
            print_log(f' current learn rate {curLR}', log_file )

        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f' current learn rate2 {curLR}', log_file )

        loss_train = train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=FiCls, mDATA_list=(SlideNames_train, FeatList_train, Label_train), ce_cri=ce_cri,
                                                   optimizer=optimizer_adam, optimizer1 =  optimizer_adam1, epoch=ii, params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, distill=params.distill_type,pooling = params.pooling_type)
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
        auc_val = test_attention_DTFD_preFeat_MultipleMean( classifier=classifier,dimReduction=dimReduction, attention=attention, UClassifier=FiCls, mDATA_list=(SlideNames_val, FeatList_val, Label_val), criterion=ce_cri,
                                                            epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, distill=params.distill_type,pooling = params.pooling_type)
        print_log(' ', log_file)
        print_log(f'>>>>>>>>>>> Test Epoch: {ii}', log_file)
        tauc = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=FiCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri,
                                                        epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, distill=params.distill_type, pooling = params.pooling_type)
        print_log(' ', log_file)

        if ii > int(params.EPOCH*0.25):
            if tauc > best_auc:
                best_epoch = ii
                best_auc = tauc
                if params.isSaveModel:
                    tsave_dict = {
                        'classifier': classifier.state_dict(),
                        'dim_reduction': dimReduction.state_dict(),
                        'attention': attention.state_dict(),
                        'att_classifier': FiCls.state_dict()
                    }
                    torch.save(tsave_dict, save_dir)

            print_log(f' test auc: {best_auc}, from epoch {best_epoch}', log_file)

        scheduler.step()
        scheduler1.step()


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier,  UClassifier, dimReduction, attention, epoch,criterion=None, params=None, f_log=None, writer=None, numGroup=3, distill='MIP',pooling='att'):

    classifier.eval()
    if pooling == 'att':
        attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list

    test_loss = AverageMeter()
    test_loss1 = AverageMeter()

    gPred = torch.FloatTensor().to(params.device)
    gt = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst] for sst in tidx_slide ]
            allSlide_pred_softmax = []

            for tidx, tfeat in enumerate(batch_feat):
                tslideLabel = label_tensor[tidx].unsqueeze(0)

                #WSI pred
                gSlidePred, afeat, AA = UClassifier(tfeat,numGroup)
                Slide_pred_softmax = torch.softmax(gSlidePred, dim=1)
                allSlide_pred_softmax.append(Slide_pred_softmax)
                gPred = torch.cat([gPred,Slide_pred_softmax],dim=0)
                gt = torch.cat([gt, tslideLabel], dim=0)
                loss = criterion(Slide_pred_softmax, tslideLabel)
                test_loss.update(loss.item(), 1)

                if distill == 'MIP':
                    patch_pred_logits = get_cam_1d(UClassifier, afeat.unsqueeze(0)).squeeze(0)  ###  cls x n
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                    _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                    topk_sub_max = sort_idx[:1].long()
                elif distill == 'MAS':
                    _, topk_sub_max = torch.max(AA,0)

                max_sub_feat = tfeat[topk_sub_max].to(params.device)
                tmidFeat = dimReduction(max_sub_feat)
                if pooling == 'att':
                    tAA = attention(tmidFeat).squeeze(0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                elif pooling == 'avg':
                    tattFeat_tensor = torch.mean(tmidFeat, dim=0, keepdim=True)
                elif pooling == 'max':
                    tattFeat_tensor, _ = torch.max(tmidFeat, dim=0)
                    tattFeat_tensor = tattFeat_tensor.unsqueeze(0)
                sub_Predict = classifier(tattFeat_tensor)

                allSlide_pred_softmax.append(torch.softmax(sub_Predict,dim=1 ))
                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred = gPred[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred, gt)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1


def train_attention_preFeature_DTFD(mDATA_list, classifier, UClassifier, dimReduction, attention, optimizer, optimizer1, epoch, ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, distill='MIP',pooling='att'):

    SlideNames_list, mFeat_list, Label_dict = mDATA_list
    UClassifier.train()
    classifier.train()
    if pooling == 'att':
        attention.train()
    dimReduction.train()

    Train_Loss = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        #每次选择批量大小
        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)
            tfeat_Group = mFeat_list[slide_idx]

            # optimization
            gSlidePred, afeat, AA = UClassifier(tfeat_Group,numGroup)

            if distill == 'MIP':
                # optimization 2
                patch_pred_logits = get_cam_1d(UClassifier, afeat.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls
                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_sub_max = sort_idx[:1].long()

            elif distill == 'MAS':
                _, topk_sub_max = torch.max(AA,0)

            max_sub = tfeat_Group[topk_sub_max].to(params.device)
            tmidFeat = dimReduction(max_sub)
            if pooling == 'att':
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            elif pooling == 'avg':
                tattFeat_tensor = torch.mean(tmidFeat, dim=0,keepdim=True)
            elif pooling == 'max':
                tattFeat_tensor, _ = torch.max(tmidFeat,dim=0)
                tattFeat_tensor = tattFeat_tensor.unsqueeze(0)

            sub_Predict = classifier(tattFeat_tensor)

            loss = ce_cri(gSlidePred, tslideLabel).mean()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer.step()

            loss1 = ce_cri(sub_Predict, tslideLabel).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            if pooling == 'att':
                torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            optimizer1.step()


            Train_Loss.update(loss.item(), 1)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss.avg}, Second Loss : {Train_Loss1.avg} '
            print_log(tstr, f_log)

    writer.add_scalar(f'train_loss ', Train_Loss.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)
    return Train_Loss.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA,numGroup = 8):

    tumorSlides = set()
    with open(testMask_dir) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if row[1] == 'Tumor':
                tumorSlides.add(row[0])

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]

        featCluter = []
        for tpatch in patch_data_list:
            tfeat = tpatch['feature']
            featCluter.append(tfeat)
        patch_data_array = np.array(featCluter)

        # K-means聚类
        cluster_metric, cluster_class = cluster_representation(patch_data_array, numGroup)
        Class = list(np.array(cluster_class))
        featGroup = [[] for _ in range(numGroup)]
        for i in range(len(Class)):
            featGroup[Class[i]].append(torch.from_numpy(featCluter[i]).unsqueeze(0))
        featGroup_tensor = []
        for j in range(numGroup):
            featGroup_tensor.append(torch.cat(featGroup[j], dim=0))
        FeatList.append(featGroup_tensor)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(mDATA,numGroup = 8):

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        # 匹配以指定字符串开头
        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]

        featCluter = []
        for tpatch in patch_data_list:
            tfeat = tpatch['feature']
            featCluter.append(tfeat)
        patch_data_array = np.array(featCluter)

        # K-means聚类
        cluster_metric, cluster_class = cluster_representation(patch_data_array,numGroup)
        Class = list(np.array(cluster_class))
        featGroup = [[] for _ in range(numGroup)]
        for i in range(len(Class)):
            featGroup[Class[i]].append(torch.from_numpy(featCluter[i]).unsqueeze(0))
        featGroup_tensor = []
        for j in range(numGroup):
            featGroup_tensor.append(torch.cat(featGroup[j],dim = 0))
        FeatList.append(featGroup_tensor)


    return SlideNames, FeatList, Label

def loder_dataSets(mDATA, numGroup = 8):

    label_csv = 'TCGA.csv'
    bags_path = pd.read_csv(label_csv)
    label_set = {}

    for i in range(len(bags_path)):
        csv_file = bags_path.iloc[i]
        slide_name = csv_file.iloc[0].split('/')[1]
        label_set[slide_name] = int(csv_file.iloc[1])

    SlideNames, FeatList, Label = [], [], []

    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)
        Label.append(label_set[slide_name])
        patch_data_list = mDATA[slide_name]

        featCluter = []
        for tpatch in patch_data_list:
            tfeat = tpatch['feature']
            featCluter.append(tfeat)
        patch_data_array = np.array(featCluter)

        # K-means聚类
        cluster_metric, cluster_class = cluster_representation(patch_data_array, numGroup)
        Class = list(np.array(cluster_class))
        featGroup = [[] for _ in range(numGroup)]
        for i in range(len(Class)):
            featGroup[Class[i]].append(torch.from_numpy(featCluter[i]).unsqueeze(0))
        featGroup_tensor = []
        for j in range(numGroup):
            if featGroup[j] !=[]:
                featGroup_tensor.append(torch.cat(featGroup[j], dim=0))
        FeatList.append(featGroup_tensor)

    return SlideNames, FeatList, Label

def cluster_representation(feature_list, num_cluster=8):
    """
    Cluster patches

    Parameters:
        - feature_list: contain a list of patch embedding and patch path

    Return:
        - labels: kmean cluster for each patch
        - cluster distance: distance of patch from centroid
    """
    kmeans = faiss.Kmeans(feature_list.shape[1], min(num_cluster, len(feature_list)),  gpu = True)
    kmeans.train(feature_list)
    label_metric, label = kmeans.assign(feature_list)
    return label_metric, label

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print('程序总用时:',time.time() - start_time)
