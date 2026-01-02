import torch
import torch.nn as nn
from ode_func_A import ODEFunc
from diffeq_solver import DiffeqSolver
from Params import args
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.n_traj_samples = int(args.n_traj_samples)
        self.ode_method = args.ode_method
        self.atol = float(args.odeint_atol)
        self.rtol = float(args.odeint_atol)
        self.gcn_step = int(args.gcn_step)
        self.filter_type = int(args.filter_type)
        self.num_gen_layer = int(args.gen_layers)
        self.ode_gen_dim = int(args.gen_dim)
        self.num_nodes = int(args.areaNum)

        ####################################################
        # FUSION PROJECTION MLP
        ####################################################
        self.latdim = int(args.latdim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.latdim * 2, self.latdim),
            nn.ReLU(),
            nn.Linear(self.latdim, self.latdim)
        )

        ####################################################
        # DECODER (after ODE evolution)
        ####################################################
        self.decoder = Decoder(
            latent_dim=self.latdim,
            cate_num=args.cateNum,   # typically 4
            num_nodes=self.num_nodes
        )

        ####################################################
        # ode solver
        ####################################################
  
        ode_set_str = "ODE setting --latent {} --samples {} --method {} \
            --atol {:6f} --rtol {:6f} --gen_layer {} --gen_dim {}".format(\
                self.latdim, self.n_traj_samples, self.ode_method, \
                self.atol, self.rtol, self.num_gen_layer, self.ode_gen_dim)
        self.odefunc = ODEFunc(
            self.ode_gen_dim, # hidden dimension
            self.latdim, 
            self.gcn_step,
            self.num_nodes,
            self.num_gen_layer,
            filter_type=self.filter_type
        ).to(device)
        self.diffeq_solver = DiffeqSolver(self.odefunc,
            self.ode_method, 
            self.latdim, 
            odeint_rtol=self.rtol,
            odeint_atol=self.atol
        )
        print(ode_set_str)
        self.save_latent = bool(args.save_latent)
        self.latent_feat = None # used to extract the latent feature

    ####################################################
    # LATENT GRAPH LEARNING FUNCTION
    ####################################################
    def compute_latent_graph(self, H, G):
        # H, G shapes coming into compute_latent_graph:
        # H: (B, C, Area, Time, Cate)
        # G: (B, C, Area, Time, Cate)

        # Reduce time+category to pooled representation
        Hn = H.mean(dim=(3, 4))      # (B, C, Area)
        Gn = G.mean(dim=(3, 4))      # (B, C, Area)

        # Move feature/channel dim to the end
        Hn = Hn.permute(0, 2, 1)     # (B, Area, C)
        Gn = Gn.permute(0, 2, 1)     # (B, Area, C)

        # Compute adjacency
        A = torch.matmul(Hn, Gn.transpose(1, 2))    # (B, Area, Area)

        return A.detach().cpu().numpy()

    def forward(self, embeds_true, neg, time_steps_to_predict):
        # print("===== TRAINING ARGS =====")
        #for k, v in vars(args).items():
        #    print(f"{k}: {v}")
        #print("==========================")
        embeds_in_global = self.dimConv_in(embeds_true.unsqueeze(1))
        DGI_neg = self.dimConv_in(neg.unsqueeze(1))
        embeds_in_local = embeds_in_global.permute(0, 3, 1, 2, 4).contiguous().view(-1, args.latdim, args.row, args.col, 4)
        spa_local1 = self.spa_cnn_local1(embeds_in_local)
        spa_local2 = self.spa_cnn_local2(spa_local1)
        spa_local2 = spa_local2.view(-1, args.temporalRange, args.latdim, args.areaNum, args.cateNum).permute(0, 2, 3, 1, 4)
        tem_local1 = self.tem_cnn_local1(spa_local2)
        tem_local2 = self.tem_cnn_local2(tem_local1)
        eb_local = tem_local2.mean(3)
        eb_tra_local = self.local_tra(tem_local2)
        out_local = self.dimConv_local(eb_local).squeeze(1)

        hy_embeds, Infomax_pred = self.Hypergraph_Infomax(embeds_in_global, DGI_neg)
        tem_global1 = self.tem_cnn_global1(hy_embeds)
        tem_global2 = self.tem_cnn_global2(tem_global1)
        tem_global3 = self.tem_cnn_global3(tem_global2)
        tem_global4 = self.tem_cnn_global4(tem_global3)
        eb_global = tem_global4.squeeze(3)
        eb_tra_global = self.global_tra(tem_global4)
        # Fix time dimension mismatch: repeat global embedding across time
        if eb_tra_global.shape[3] == 1:
            eb_tra_global = eb_tra_global.repeat(1, 1, 1, args.temporalRange, 1)

        out_global = self.dimConv_global(eb_global).squeeze(1)

        #print("eb_tra_global shape:", eb_tra_global.shape)
        #print("eb_tra_local shape:", eb_tra_local.shape)

        # ============================================================
        # NEW SECTION: LATENT ADJACENCY + FUSION + ODE
        # ============================================================

        # ---------- STEP 1: Identify H and gamma ----------
        # H = eb_tra_local : (B, C, A, T, K)
        # γ = eb_tra_global: (B, C, A, T, K)
        # STEP 1: Pool temporal + category
        H = eb_tra_local.mean(dim=(3,4))   # (B, C, Area)
        G = eb_tra_global.mean(dim=(3,4))  # (B, C, Area)

        # STEP 2: (B, Area, C)
        H = H.permute(0, 2, 1)
        G = G.permute(0, 2, 1)

        # STEP 3: latent adjacency (B, Area, Area)
        A_latent = torch.matmul(H, G.transpose(1,2))
        # print("A_latent shape:", A_latent.shape)

        # ---------- STEP 3: Set adjacency inside ODEFunc ----------
        self.odefunc.set_adjacency(A_latent.detach())

        # ---------- STEP 4: Fuse H + G → Zₜ₀ ----------
        fusion = torch.cat([H, G], dim=-1)       # (B, N, 2C)
        Z_t0 = self.fuse_mlp(fusion)             # (B, N, C)
        B = Z_t0.shape[0]
        # reshape for ODE solver: (n_samples, B, num_nodes * C)
        Z_t0 = Z_t0.reshape(B, self.num_nodes * self.latdim)
        Z_t0 = Z_t0.unsqueeze(0).repeat(self.n_traj_samples, 1, 1)

        # ---------- STEP 5: Solve the ODE ----------
        sol_ys, fe = self.diffeq_solver(Z_t0, time_steps_to_predict)
        # sol_ys: (T, n_samples, B, num_nodes * C)

        # --------- STEP 6: Extract last-step latent state ----------
        Z_t = sol_ys[-1]        # (n_samples, B, num_nodes * C)
        Z_t = Z_t.mean(0)       # (B, num_nodes * C)

        # ---------- STEP 7: Decode Z(t*) to predictions ----------
        pred = self.decoder(Z_t)   # (B, AreaNum, CateNum)

        return out_local, eb_tra_local, eb_tra_global, Infomax_pred, out_global, fe, Z_t, pred

class Decoder(nn.Module):
    def __init__(self, latent_dim, cate_num, num_nodes):
        super().__init__()
        self.latent_dim = latent_dim
        self.cate_num = cate_num
        self.num_nodes = num_nodes

        # Node-wise predictor: C → 4 crime categories
        self.linear = nn.Linear(latent_dim, cate_num)

    def forward(self, Z):
        """
        Z: (B, num_nodes * latent_dim)
        return: (B, num_nodes, cate_num)
        """

        B = Z.size(0)

        # reshape into per-node latent embeddings
        Z = Z.reshape(B, self.num_nodes, self.latent_dim)   # (B, N, C_latent)

        # apply node-wise prediction
        pred = self.linear(Z)       

        return pred
