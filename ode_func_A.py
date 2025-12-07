# ==============================
# UPDATED ODEFunc.py
# ==============================

import numpy as np
import torch
import torch.nn as nn
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(f'{self._type}_weight_{shape}', nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(f'{self._type}_biases_{length}', biases)
        return self._biases_dict[length]


class ODEFunc(nn.Module):
    def __init__(self, num_units, latent_dim, gcn_step, num_nodes,
                 gen_layers=1, nonlinearity='tanh', filter_type="default"):
        """
        Updated ODEFunc: adjacency no longer passed in __init__.
        Latent adjacency A(t) is provided dynamically using set_adjacency().
        """
        super(ODEFunc, self).__init__()

        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        self._num_nodes = num_nodes
        self._num_units = num_units
        self._latent_dim = latent_dim
        self._gen_layers = gen_layers
        self._gcn_step = gcn_step
        self._filter_type = filter_type

        self._supports = None              # <-- latent adjacency (supports) set dynamically
        self._gconv_params = LayerParams(self, 'gconv')
        
        # -----------------------------
        # EAGER CREATION OF PARAMETERS
        # -----------------------------
        num_matrices = 2 * self._gcn_step + 1   # because supports = [A, A^T]

        # shapes for latent → latent and latent → units transforms
        in_lat   = num_matrices * self._latent_dim
        in_units = num_matrices * self._num_units

        # (1) for _gconv(inputs, latent_dim)
        self._gconv_params.get_weights((in_lat, self._latent_dim))
        self._gconv_params.get_biases(self._latent_dim)

        # (2) for _gconv(c, num_units)
        self._gconv_params.get_weights((in_lat, self._num_units))
        self._gconv_params.get_biases(self._num_units)

        # (3) for _gconv(c, latent_dim) at end of gen_layers
        self._gconv_params.get_weights((in_units, self._latent_dim))
        # biases for latent_dim already created

        self.nfe = 0

        # ----------------------------------------
        # NEW METHOD FOR SETTING LATENT ADJACENCY
        # ----------------------------------------
    def set_adjacency(self, A_latent):
        """
        A_latent: (B, N, N) or (N, N)
        Updated adjacency for ODEFunc.
        """

        # --- STEP 1: collapse batch dimension (if exists) ---
        if isinstance(A_latent, torch.Tensor):
            if A_latent.dim() == 3:
                # average adjacency across batch
                A_latent = A_latent.mean(dim=0)

            # detach so it doesn't require grad, move to CPU for numpy ops
            A_latent_np = A_latent.detach().cpu().numpy()

        elif isinstance(A_latent, np.ndarray):
            A_latent_np = A_latent
        else:
            raise ValueError("A_latent must be torch.Tensor or numpy array")

        # --- STEP 2: compute random walk matrices ---
        supports = [
            utils.calculate_random_walk_matrix(A_latent_np).T,
            utils.calculate_random_walk_matrix(A_latent_np.T).T,
        ]

        # --- STEP 3: convert them into PyTorch sparse matrices ---
        self._supports = []
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))


    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        idx = np.column_stack((L.row, L.col))
        idx = idx[np.lexsort((idx[:, 0], idx[:, 1]))]
        return torch.sparse_coo_tensor(idx.T, L.data, L.shape, device=device)

    # ----------------------------------------------------------------
    # FORWARD: essential ODE function (computes dZ/dt)
    # ----------------------------------------------------------------
    def forward(self, t_local, y, backwards=False):
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, y)
        return -grad if backwards else grad

    # ----------------------------------------------------------------
    # Compute gradient for ODE
    # ----------------------------------------------------------------
    def get_ode_gradient_nn(self, t_local, inputs):
        if self._filter_type == "unkP":
            return self._fc(inputs)

        if self._filter_type == "IncP":
            return -self.ode_func_net(inputs)

        # Default diffusion model
        theta = torch.sigmoid(self._gconv(inputs, self._latent_dim, bias_start=1.0))
        return -theta * self.ode_func_net(inputs)

    # ODE NN model
    def ode_func_net(self, inputs):
        c = inputs
        for _ in range(self._gen_layers):
            c = self._gconv(c, self._num_units)
            c = self._activation(c)

        c = self._gconv(c, self._latent_dim)
        return self._activation(c)

    # FC case (not used in your pipeline)
    def _fc(self, inputs):
        B = inputs.size(0)
        grad = self.gradient_net(inputs.view(B * self._num_nodes, self._latent_dim))
        return grad.view(B, self._num_nodes * self._latent_dim)

    # --------------------------------------------------------------
    # Graph convolution with learnable filters
    # --------------------------------------------------------------
    def _gconv(self, inputs, output_size, bias_start=0.0):
        if self._supports is None:
            raise ValueError("Adjacency not set. Call set_adjacency(A_latent) before ODE solve.")

        B = inputs.shape[0]
        inputs = inputs.reshape(B, self._num_nodes, -1)
        x0 = inputs.permute(1, 2, 0).reshape(self._num_nodes, -1)
        x = x0.unsqueeze(0)

        # multi-step propagation
        if self._gcn_step > 0:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)

                for _ in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._gcn_step + 1
        x = x.reshape(num_matrices, self._num_nodes, -1, B).permute(3, 1, 2, 0)
        x = x.reshape(B * self._num_nodes, -1)

        W = self._gconv_params.get_weights((x.size(1), output_size))
        b = self._gconv_params.get_biases(output_size, bias_start)

        x = torch.matmul(x, W) + b
        return x.reshape(B, self._num_nodes * output_size)
