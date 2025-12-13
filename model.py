import torch
import torch.nn as nn
from Params import args
from ode_func_A import ODEFunc
from diffeq_solver import DiffeqSolver
from Params import args
import utils

# Local
# Local Spatial cnn
class spa_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(spa_cnn_local, self).__init__()
        self.spaConv1 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv2 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv3 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv4 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.drop = nn.Dropout(args.dropRateL)
        self.act_lr = nn.LeakyReLU()

    def forward(self, embeds):
        cate_1 = self.drop(self.spaConv1(embeds))
        cate_2 = self.drop(self.spaConv2(embeds))
        cate_3 = self.drop(self.spaConv3(embeds))
        cate_4 = self.drop(self.spaConv4(embeds))
        spa_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(spa_cate + embeds)

# Local Temporal cnn
class tem_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(tem_cnn_local, self).__init__()
        self.temConv1 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv2 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv3 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv4 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.act_lr = nn.LeakyReLU()
        self.drop = nn.Dropout(args.dropRateL)

    def forward(self, embeds):
        cate_1 = self.drop(self.temConv1(embeds))
        cate_2 = self.drop(self.temConv2(embeds))
        cate_3 = self.drop(self.temConv3(embeds))
        cate_4 = self.drop(self.temConv4(embeds))
        tem_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(tem_cate + embeds)


# Global
# Global Hypergraph
class Hypergraph(nn.Module):
    def __init__(self):
        super(Hypergraph, self).__init__()
        self.adj = nn.Parameter(torch.Tensor(torch.randn([args.temporalRange, args.hyperNum, args.areaNum * args.cateNum])), requires_grad=True)
        self.Conv = nn.Conv3d(args.latdim, args.latdim, kernel_size=1)
        self.act1 = nn.LeakyReLU()

    def forward(self, embeds):
        adj = self.adj
        tpadj = adj.transpose(2, 1)
        embeds_cate = embeds.transpose(2, 3).contiguous().view(embeds.shape[0], args.latdim, args.temporalRange, -1)
        hyperEmbeds = self.act1(torch.einsum('thn,bdtn->bdth', adj, embeds_cate))
        retEmbeds = self.act1(torch.einsum('tnh,bdth->bdtn', tpadj, hyperEmbeds))
        retEmbeds = retEmbeds.view(embeds.shape[0], args.latdim, args.temporalRange, args.areaNum, args.cateNum).transpose(2, 3)
        return retEmbeds

# Hypergraph Infomax AvgReadout
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, embeds):
        return torch.mean(embeds, 2)

# Hypergraph Infomax Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(args.latdim, args.latdim, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, score, h_pos, h_neg):
        score = torch.unsqueeze(score, 2)
        score = score.expand_as(h_pos)
        score = score.transpose(1, 4).contiguous()
        h_pos = h_pos.transpose(1, 4).contiguous()
        h_neg = h_neg.transpose(1, 4).contiguous()
        sc_pos = torch.squeeze(self.f_k(h_pos, score), -1)
        sc_neg = torch.squeeze(self.f_k(h_neg, score), -1)
        logits = torch.cat((sc_pos.mean(-1), sc_neg.mean(-1)), dim=2)
        return logits

# Global Hypergraph Infomax
class Hypergraph_Infomax(nn.Module):
    def __init__(self):
        super(Hypergraph_Infomax, self).__init__()
        self.Hypergraph = Hypergraph()
        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator()

    def forward(self, eb_pos, eb_neg):
        h_pos = self.Hypergraph(eb_pos)
        c = self.readout(h_pos)
        score = self.sigm(c)
        h_neg = self.Hypergraph(eb_neg)
        ret = self.disc(score, h_pos, h_neg)
        return h_pos, ret
    
    def encode(self, eb):
        return self.Hypergraph(eb)

# =========================
# Ψ discriminator on Z(t)
# =========================
class ZDiscriminator(nn.Module):
    def __init__(self, feat_dim):
        super(ZDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(feat_dim, feat_dim, 1)

        torch.nn.init.xavier_uniform_(self.f_k.weight)
        if self.f_k.bias is not None:
            self.f_k.bias.data.fill_(0.0)

    def forward(self, c, h_pos, h_neg):
        """
        c:     (B, F)
        h_pos: (B, N, F)
        h_neg: (B, N, F)
        """
        # Expand summary vector → (B, N, F)
        c_expanded = c.unsqueeze(1).expand_as(h_pos)

        sc_pos = self.f_k(h_pos, c_expanded).squeeze(-1)   # (B, N)
        sc_neg = self.f_k(h_neg, c_expanded).squeeze(-1)   # (B, N)

        # logits = [mean positive score , mean negative score]
        logits = torch.stack([sc_pos.mean(1), sc_neg.mean(1)], dim=1)
        return logits

# ============================================
# NEW Z-Infomax (Ψ) for (B, N, F)
# ============================================
class ZInfomax(nn.Module):
    def __init__(self, feat_dim=args.gen_dim):
        super(ZInfomax, self).__init__()
        self.sigm = nn.Sigmoid()
        self.disc = ZDiscriminator(feat_dim)

    def forward(self, z_pos, z_neg):
        """
        z_pos, z_neg: (B, N, F)
        """
        # readout = mean over nodes → (B, F)
        c = z_pos.mean(1)
        score = self.sigm(c)

        return self.disc(score, z_pos, z_neg)

# Global Temporal cnn
class tem_cnn_global(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(tem_cnn_global, self).__init__()
        self.kernel_size = kernel_size
        self.temConv = nn.Conv3d(input_dim, output_dim, kernel_size=[1, kernel_size, 1], stride=1, padding=[0,0,0])
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(args.dropRateG)

    def forward(self, embeds):
        ret_flow = self.temConv(embeds)
        ret_drop = self.drop(ret_flow)
        return self.act(ret_drop)

# embedding transform
class Transform_3d(nn.Module):
    def __init__(self):
        super(Transform_3d, self).__init__()
        self.BN = nn.BatchNorm3d(args.latdim)
        self.Conv1 = nn.Conv3d(args.latdim, args.latdim, kernel_size=1)

    def forward(self, embeds):
        embeds_BN = self.BN(embeds)
        embeds1 = self.Conv1(embeds_BN)
        return embeds1


class STHSL(nn.Module):
    def __init__(self):
        super(STHSL, self).__init__()

        self.dimConv_in = nn.Conv3d(1, args.latdim, kernel_size=1, padding=0, bias=True)
        self.dimConv_local = nn.Conv2d(args.latdim, 1, kernel_size=1, padding=0, bias=True)
        self.dimConv_global = nn.Conv2d(args.latdim, 1, kernel_size=1, padding=0, bias=True)

        self.spa_cnn_local1 = spa_cnn_local(args.latdim, args.latdim)
        self.spa_cnn_local2 = spa_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local1 = tem_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local2 = tem_cnn_local(args.latdim, args.latdim)

        self.Hypergraph_Infomax = Hypergraph_Infomax()
        self.tem_cnn_global1 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global2 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global3 = tem_cnn_global(args.latdim, args.latdim, 9)
        self.tem_cnn_global4 = tem_cnn_global(args.latdim, args.latdim, 6)

        self.local_tra = Transform_3d()
        self.global_tra = Transform_3d()

        # MLP to fuse (local_last + global)
        self.fusion_mlp = nn.Sequential(
            nn.Conv3d(2 * args.latdim, args.latdim, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(args.latdim, args.gen_dim, kernel_size=1)
        )

        # adjacency
        adj_mx = utils.build_crime_adj()

        # print("DEBUG ADJ SHAPE:", adj_mx.shape)

        self.num_nodes = args.areaNum * args.cateNum

        # DE-net ODEFunc
        self.odefunc = ODEFunc(
            num_units=args.gen_dim,
            latent_dim=args.gen_dim,
            adj_mx=adj_mx,
            gcn_step=args.gcn_step,
            num_nodes=self.num_nodes,
            gen_layers=1,
            nonlinearity='tanh',
            filter_type="default"
        )

        # Solver
        self.de_solver = DiffeqSolver(
            self.odefunc,
            method=args.ode_method,
            latent_dim=args.gen_dim
        )

        # Readout to crime prediction
        self.de_readout = nn.Linear(args.gen_dim, 1)

        # Ψ discriminator acts on Z(t)
        self.psi = ZInfomax(args.gen_dim)

    def forward(self, embeds_true, neg):
        embeds_in_global = self.dimConv_in(embeds_true.unsqueeze(1))
        DGI_neg = self.dimConv_in(neg.unsqueeze(1))
        embeds_in_local = embeds_in_global.permute(0, 3, 1, 2, 4).contiguous().view(-1, args.latdim, args.row, args.col, 4)

        # ====== local pattern evolution ======
        spa_local1 = self.spa_cnn_local1(embeds_in_local)
        spa_local2 = self.spa_cnn_local2(spa_local1)
        spa_local2 = spa_local2.view(-1, args.temporalRange, args.latdim, args.areaNum, args.cateNum).permute(0, 2, 3, 1, 4)
        tem_local1 = self.tem_cnn_local1(spa_local2)
        tem_local2 = self.tem_cnn_local2(tem_local1)
        eb_local = tem_local2.mean(3)
        eb_tra_local = self.local_tra(tem_local2)
        out_local = self.dimConv_local(eb_local).squeeze(1)

        # ====== global pattern evolution ======
        hy_embeds = self.Hypergraph_Infomax.encode(embeds_in_global)
        
        # hy_embeds, Infomax_pred = self.Hypergraph_Infomax(embeds_in_global, DGI_neg)
        tem_global1 = self.tem_cnn_global1(hy_embeds)
        tem_global2 = self.tem_cnn_global2(tem_global1)
        tem_global3 = self.tem_cnn_global3(tem_global2)
        tem_global4 = self.tem_cnn_global4(tem_global3)
        eb_global = tem_global4.squeeze(3)
        eb_tra_global = self.global_tra(tem_global4)

        # ====== FUSION FOR Z0 ======
        # local last timestep: (B, C, A, 1, K)
        eb_local_last = eb_tra_local[:, :, :, -1:, :]

        # fusion = concat(local_last, global)
        fusion = torch.cat([eb_local_last, eb_tra_global], dim=1)

        # (B, C_z, A, 1, K)
        z0 = self.fusion_mlp(fusion)

        # ====== ODE EVOLUTION ======
        B, C_z, A, _, K = z0.shape
        # print("DEBUG z0:", z0.shape, "nodes:", A*K, "expected:", self.num_nodes)
        # 1) (B, C_z, A, K)
        z0 = z0.squeeze(3)

        # 2) (B, A, K, C_z)
        z0 = z0.permute(0, 2, 3, 1).contiguous()

        # 3) (B, A*K, C_z)
        z0 = z0.reshape(B, A * K, C_z)

        # print("DEBUG Z0 (after reshape):", z0.shape)

        # 4) Flatten for ODEFunc  → (1, B, 256*C_z)
        z0_flat = z0.reshape(B, -1).unsqueeze(0)

        # time steps for ODE
        t = torch.tensor([0.0], device=z0.device)

        # old option1
        # t = torch.linspace(0, args.horizon, steps=args.horizon+1).to(z0.device)

        sol, _ = self.de_solver(z0_flat, t)
        Zt = sol[-1, 0].view(B, self.num_nodes, C_z)  # (B, N, C_z)

        # ====== Ψ DISCRIMINATOR ON Z(t) ======
        idx = torch.randperm(B, device=Zt.device)
        Zt_neg = Zt[idx].clone()
        Infomax_pred = self.psi(Zt, Zt_neg)

        # ====== DE-NET PREDICTION ======
        pred_nodes = self.de_readout(Zt)      # (B, N, 1)
        out_global = pred_nodes.view(B, A, K) # same shape as old out_global

        return out_local, eb_tra_local, eb_tra_global, Infomax_pred, out_global, Zt