import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from models.pointnet import PointNetEncoder
import sys
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion
sys.path.append("./emd/")
import emd_module as emd


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))

        return x

class CycleNet(nn.Module):
    def __init__(self, num_class=40, num_out_points=1024, bottleneck_size=1024, n_primitives=16):
        super(CycleNet, self).__init__()
        self.num_out_points = num_out_points
        self.bottleneck_size = bottleneck_size + num_class
        self.n_primitives = n_primitives

        # PointNet encoder
        self.feat_partial = PointNetEncoder(global_feat=True, feature_transform=True) 
        self.feat_recon = PointNetEncoder(global_feat=True, feature_transform=True)
        
        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        # Decoder
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.expansion = expansion.expansionPenaltyModule()


    def forward(self, x, input_mode):
        # Prediction branch
        if input_mode == "partial":
            global_feat, _, _ = self.feat_partial(x) # global_feat: (B, 1024)
        elif input_mode == "recon":
            global_feat, _, _ = self.feat_recon(x) # global_feat: (B, 1024)
        pred_cls = F.relu(self.bn1(self.fc1(global_feat)))
        pred_cls = F.relu(self.bn2(self.dropout(self.fc2(pred_cls))))
        pred_cls = self.fc3(pred_cls)
        pred_cls = F.log_softmax(pred_cls, dim=1)

        ## Turn into one hot vector 
        pred_choice = pred_cls.data.max(1)[1].unsqueeze(1).cpu()
        y_one_hot = torch.zeros(pred_cls.size()).scatter_(1, pred_choice, 1).cuda()
        # Reconstruction brach
        condition_feat = torch.cat((global_feat, y_one_hot), 1)

        # Reconstruction brach
        #condition_feat = torch.cat((global_feat, pred_cls), 1)

        patch_outs = []
        for i in range(0,self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(condition_feat.size(0), 2, self.num_out_points//self.n_primitives)) # rand_grid: (B, 2, 512)
            rand_grid.data.uniform_(0,1)
            patch = condition_feat.unsqueeze(2).expand(condition_feat.size(0), condition_feat.size(1), rand_grid.size(2)).contiguous() # patch: (B, 1024 + num_class, 512)
            patch = torch.cat( (rand_grid, patch), 1).contiguous() # patch: (B, 1026 + num_class, 512)
            patch_outs.append(self.decoder[i](patch)) # patch_outs: (16, B, 3, 512)

        patch_outs = torch.cat(patch_outs, 2).contiguous() # patch_outs: (B, 3, num_out_points)
        recon_output = patch_outs.transpose(1, 2).contiguous() # recon_output: (B, num_out_points, 3)

	# Compute expansion loss (using Minimum Spanning Tree (MST))
        if self.n_primitives < 16:
            loss_mst = torch.tensor(0).float().cuda()
        else:
            dist, _, mean_mst_dis = self.expansion(recon_output, self.num_out_points//self.n_primitives, 1.5)
            loss_mst = torch.mean(dist)

        return pred_cls, recon_output, loss_mst
        

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.EMD_criterion = emd.emdModule()

    def forward(self, pred, target, recon_output, complete, eps, iters):
        # Compute cls loss
        loss_cls = F.nll_loss(pred, target)

        # Compute EMD loss
        dist, _ = self.EMD_criterion(recon_output, complete, eps, iters)
        loss_EMD = torch.sqrt(dist).mean(1)

        return loss_cls, loss_EMD
