import numpy as np
import sys, random, os, glob, torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from data_loader import create_dataset, create_dataloader, create_datasetTest
from torchvision.utils import save_image
from tqdm import tqdm
from archs import vggfeat
from util_loss import ContrasLoss
from thop import profile

sys.path.append('../../')
sys.path.append('.')  # for vscode debug
sys.path.append('/home/autohdr/codes/DSINE/projects/dsine/NAFNet')
import utils.utils as utils
import projects.dsine.config as config
from utils.projection import intrins_from_fov, intrins_from_txt
from vggModel2025 import  Abvgg13bn_unetV2, VGG_SG_V2
from archs.vqgan_arch import VQGAN
from NAFNet import NAFNet

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_normal_255(normal):
    new_normal = normal * 128 + 128
    new_normal = new_normal.clamp(0, 255) / 255
    return new_normal

def unnormalize(tensor):
    """反归一化，将 tensor 从 [-1,1] 或 [0,1] 变回原始值"""
    # 归一化参数
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)  # 适配 (C, H, W) 维度
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    tensor = tensor * std + mean  # 反归一化
    return torch.clamp(tensor, 0, 1)  # 限制在 [0,1] 之间，防止溢出

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    device = torch.device('cuda:0')

    args = config.get_args(test=True)
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    seed_everything(seed=1234)

    Wcon = 0.5
    Wfeat = 0.1
    batchsize = 8
    trainloop = 500001
    start_iter = 0
    expName2025 = 'Rec_con_feat' + str(Wcon) + '_' + str(Wfeat)
    savepath = './results/VGGV2/'
    Scpkt = None
    Scpkt = '/home/autohdr/codes/DSINE/projects/dsine/results/VGGV2/Rec_con_feat0.5_0.1/exp/210000.pt'


    savepath = savepath + expName2025 
    logPath = savepath + '/' + 'log' + '/'
    mkdirss(logPath)
    imgsPath = savepath + '/' + 'imgs' + '/'
    mkdirss(imgsPath)
    cpktPath = savepath + '/' + 'exp' + '/'
    mkdirss(cpktPath)

    GetVGGFeat = vggfeat.GetVGGFeat().to(device)
    for param in GetVGGFeat.parameters():
        param.requires_grad = False
    GetVGGFeat.eval()

    TeacherModel = DSINE(args).to(device)
    TeacherModel = utils.load_checkpoint(args.ckpt_path, TeacherModel)
    TeacherModel.eval()

    StudentModel = VGG_SG_V2().to(device) 

    if Scpkt != None:
        ckptModel = torch.load(Scpkt)
        StudentModel.load_state_dict(ckptModel['StudentModel'])
        start_iter = ckptModel['iter']
        print('****************************')
        print('****************************')
        print('****************************')
        print('Load pretrained paramters.')
        print('****************************')
        print('****************************')
        print('****************************')

    g_optim = torch.optim.Adam(StudentModel.parameters(),  lr=0.00001, betas=(0.9, 0.99))
    
    DataPaths = ['/home/autohdr/Dataset/gghead/FFHQ_png_512/', '/home/autohdr/Dataset/X2_sub/', '/home/autohdr/Dataset/HR/']
    pathd = DataPaths
    train_set = create_dataset(pathd, dataflag='MixData', phase='train')
    train_dl = create_dataloader(train_set, batchsize, 'train')

    DataPaths = ['/home/autohdr/Dataset/DIV2K_valid_HR_sub_S']
    pathd = DataPaths
    test_set = create_datasetTest(pathd, dataflag='WoRandom', phase='test')
    test_dl = create_dataloader(test_set, 8, 'Val')

    train_loader = sample_data(train_dl)
    test_loader = sample_data(test_dl)

    CosLoss = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    # 初始化 ContrasLoss
    contras_loss_fn = ContrasLoss(loss_weight=Wcon, weights=[1.0]).to(device)
    pbar = range(trainloop)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)


    x = torch.rand((1, 3, 256, 256)).cuda()
    flops_1, params_1 = profile(StudentModel, inputs=(x, ))
    print('StudentModelFLOPs = ' + str((flops_1)/1000**3) + 'G')
    print('StudentModelParams = ' + str((params_1)/1000**2) + 'M')
    
    flops_1, params_1 = profile(TeacherModel, inputs=(x, ))
    print('TeacherModelFLOPs = ' + str((flops_1)/1000**3) + 'G')
    print('TeacherModelParams = ' + str((params_1)/1000**2) + 'M')
    

    for idx in pbar:
        iter = idx + start_iter
        if iter > trainloop:
            print("Training is finished!")
            break

        inputImgs, index = next(train_loader)
        inputImgs = inputImgs.to(device)


        with torch.no_grad():
            b, c, orig_H, orig_W = inputImgs.shape
            lrtb = utils.get_padding(orig_H, orig_W)
            intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

            intrins[:, 0, 2] += lrtb[0]
            intrins[:, 1, 2] += lrtb[2]
            intrins_ex = intrins.clone().expand([b, 3, 3])

            pred_norm = TeacherModel(inputImgs, intrins=intrins_ex)[-1]
            pred_norm = F.normalize(pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]).detach()


        refine_norm = F.normalize(StudentModel(inputImgs))
        loss_rec = 1 - CosLoss(refine_norm, pred_norm).mean()

        features_A = GetVGGFeat(unnormalize(inputImgs))
        features_B = GetVGGFeat(refine_norm/2 + 0.5)

        geo_loss1 = F.l1_loss(features_A[0], features_B[0])
        geo_loss2 = F.l1_loss(features_A[1], features_B[1])
        geo_loss3 = F.l1_loss(features_A[2], features_B[2]) 

        geo_loss = Wfeat * (geo_loss1 + geo_loss2 + geo_loss3)

        features_T = GetVGGFeat(pred_norm)[-1]
        features_S = GetVGGFeat(refine_norm)[-1]
        fb, fc, fw, fh = features_T.shape

        features_T = features_T.view([fb, fc, -1])
        features_S = features_S.view([fb, fc, -1])

        # 生成两个负样本
        neg1 = features_T + 0.1 * torch.randn_like(features_T)
        # neg2 = features_T[torch.randperm(features_T.shape[1])].unsqueeze(0)

        # 打包 latents   需要的格式： [anc, pos, neg1, neg2, ...]
        latents = [[features_T, features_S, neg1]]

        # 计算损失
        loss_con = Wcon * contras_loss_fn(latents)

        LossTotal = loss_rec + loss_con + geo_loss

        g_optim.zero_grad()
        LossTotal.backward()
        g_optim.step()

        pbar.set_description(
            (
                f"iter:{iter:6d}; idx:{idx:6d}; loss_rec:{loss_rec.item():.4f}; loss_con:{loss_con.item():.10f}; geo_loss:{geo_loss.item():.10f};"
            )
        )


        if iter % 5001 == 0 and iter!=0:
            with torch.no_grad():
                for ii, batch in enumerate(test_dl):
                    inputImgs, labels = batch
                    inputImgs = inputImgs.to(device)
                    b, c, orig_H, orig_W = inputImgs.shape
                    lrtb = utils.get_padding(orig_H, orig_W)
                    intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

                    intrins[:, 0, 2] += lrtb[0]
                    intrins[:, 1, 2] += lrtb[2]
                    intrins_ex = intrins.clone().expand([b, 3, 3])

                    pred_norm = TeacherModel(inputImgs, intrins=intrins_ex)[-1]
                    pred_norm = F.normalize(pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]).detach()

                    refine_norm = F.normalize(StudentModel(inputImgs))

                    svgImg = torch.cat([unnormalize(inputImgs), get_normal_255(pred_norm), get_normal_255(refine_norm)])
                    save_image(svgImg, imgsPath + '%d'%(iter) + '_testing.png', nrow=b, normalize=False)
                    break
    
        if iter % 500 == 0 and iter!=0:
            svgImg = torch.cat([unnormalize(inputImgs), get_normal_255(pred_norm), get_normal_255(refine_norm)])
            save_image(svgImg, imgsPath + '%d'%(888) + '_training.png', nrow=b, normalize=False)

        if iter % 5000 == 0 and iter!=0:
            torch.save(
            {
                "StudentModel": StudentModel.state_dict(),
                "iter": iter,
            },
            f"%s/{str(iter).zfill(6)}.pt"%(cpktPath),
            )
