import numpy as np
import torch
import dgl
import random
import warnings
import networkx as nx


from scipy.sparse import csr_matrix, coo_matrix, spdiags, eye
from scipy.linalg import svd

class NegativeSampleGenerator:
    def __init__(self,
                 perturbation_methods=('feature_swap','feature_noise','structure'),
                 attr_swap_ratio=0.2,
                 spectral_perturb_ratio=0.2,  # deprecated
                 feature_noise_std=0.1,
                 threshold=0.5,
                 weighted_graph=False,
                 verbose=False,
                 big_energy_threshold=0.5,
                 mid_energy_threshold=0.75,
                 max_ratio=10.0,
                 spco_theta=1.0,
                 spco_epsilon=0.1,
                 spco_lam=0.5,
                 spco_sinkhorn_iter=3
                 ):
        """
        ----
        perturbation_methods: list of str
         , ['feature_swap', 'feature_noise', 'structure']
        attr_swap_ratio: float

        spectral_perturb_ratio: float

        feature_noise_std: float

        threshold: float

        weighted_graph: bool

        verbose: bool

        big_energy_threshold: float

        max_ratio: float

        """
        self.perturbation_methods = perturbation_methods
        self.attr_swap_ratio = attr_swap_ratio
        self.spectral_perturb_ratio = spectral_perturb_ratio  # deprecated 
        self.feature_noise_std = feature_noise_std
        self.threshold = threshold
        self.weighted_graph = weighted_graph
        self.verbose = verbose

        self.big_energy_threshold = big_energy_threshold
        self.mid_energy_threshold = mid_energy_threshold
        self.max_ratio = max_ratio
        
        # SpCo
        self.spco_theta = spco_theta
        self.spco_epsilon = spco_epsilon
        self.spco_lam = spco_lam
        self.spco_sinkhorn_iter = spco_sinkhorn_iter
        
        

    def generate_negative_graphs(self, g):
        """

        """
        g_add = self.generate_negative_graph(g, increase_edges=True)
        g_remove = self.generate_negative_graph(g, increase_edges=False)
        return g_add, g_remove

    def generate_negative_graph(self, g, increase_edges=True):
        """
        """
        g_neg = g.clone()


        if any(m in ['feature_swap', 'feature_noise'] for m in self.perturbation_methods):
            g_neg = self.feature_perturbation(g_neg)

        if 'structure' in self.perturbation_methods:
            g_neg = self.structure_perturbation_svd(g_neg, increase_edges)
                       

        if 'spco' in self.perturbation_methods:
            g_neg = self.structure_perturbation_spco(g_neg, increase_edges)

        g_neg = dgl.remove_self_loop(g_neg)
        g_neg = dgl.add_self_loop(g_neg)

        if self.verbose:
            print(f"Generated negative graph: {g_neg.number_of_edges()} edges (increase={increase_edges})")

        return g_neg

    def feature_perturbation(self, g):
        """

        """
        g_neg = g.clone()
        if 'attr' not in g_neg.ndata:
            return g_neg

        features = g_neg.ndata['attr'].clone()  # Do not mutate the normal reference.
        num_nodes = features.size(0)


        if 'feature_swap' in self.perturbation_methods:
            num_swaps = int(num_nodes * self.attr_swap_ratio)
            if num_swaps > 1:
                idx = torch.randperm(num_nodes)[:num_swaps]
                swapped_idx = torch.roll(idx, shifts=1)
                features[idx] = features[swapped_idx]


        if 'feature_noise' in self.perturbation_methods:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise

        g_neg.ndata['attr'] = features
        return g_neg

    def structure_perturbation_svd(self, g, increase_edges=True):
        """

        """
        try:
            src, dst = g.edges()
            A = np.zeros((g.num_nodes(), g.num_nodes()), dtype=np.float32)
            np.add.at(A, (src.cpu().numpy(), dst.cpu().numpy()), 1)
        except Exception as e:
            warnings.warn(f"[WARN] dense: {e}")
            return g.clone()


        try:
            U, S, Vt = svd(A, full_matrices=False)
        except Exception as e:
            warnings.warn(f"[WARN] SVD: {e}")
            return g.clone()

        S_new = self.spectral_perturbation(S, increase_edges)

        A_perturbed = (U * S_new) @ Vt
        A_perturbed = 0.5*(A_perturbed + A_perturbed.T)
        A_perturbed[A_perturbed<0] = 0


        if not self.weighted_graph:
            A_perturbed = (A_perturbed>self.threshold).astype(float)
        else:
            A_perturbed = np.clip(A_perturbed, 0, 1)

        A_sp = csr_matrix(A_perturbed)
        src, dst = A_sp.nonzero()
        g_new = dgl.graph((torch.tensor(src), torch.tensor(dst)),
                          num_nodes=A.shape[0])

        if self.weighted_graph:
            g_new.edata['weight'] = torch.tensor(A_sp.data, dtype=torch.float32)

        if 'attr' in g.ndata:
            g_new.ndata['attr'] = g.ndata['attr'].clone()

        return g_new

    def spectral_perturbation(self, S, increase_edges=True):
        """

        """

        sorted_idx_desc = np.argsort(-S)  
        S_desc = S[sorted_idx_desc]
        cum_energy = np.cumsum(S_desc**2)
        total_energy = cum_energy[-1] if len(cum_energy)>0 else 1e-9


        def find_cut_index(thres):

            for i in range(len(cum_energy)):
                if cum_energy[i]/total_energy >= thres:
                    return i+1  
            return len(cum_energy)


        big_cut = find_cut_index(self.big_energy_threshold)    
        mid_cut = find_cut_index(self.mid_energy_threshold)    

        seg_big   = np.arange(0, big_cut)
        seg_mid   = np.arange(big_cut, mid_cut)
        seg_small = np.arange(mid_cut, len(S_desc))

 
        mean_big   = S_desc[seg_big].mean()   if len(seg_big)>0 else 1.0
        mean_small = S_desc[seg_small].mean() if len(seg_small)>0 else 1.0


        ratio_raw = mean_big/(mean_small+1e-10)
        ratio_clamped = min(ratio_raw, self.max_ratio)

        if self.verbose:
            print(f" [EnergySplit] big=[0..{big_cut-1}], mid=[{big_cut}..{mid_cut-1}], small=[{mid_cut}..end]")
            print(f"   mean_big={mean_big:.4f}, mean_small={mean_small:.4f}, ratio={ratio_raw:.4f} => clamp={ratio_clamped:.4f}")


        S_new = S.copy()

        seg_big_orig   = sorted_idx_desc[seg_big]
        seg_mid_orig   = sorted_idx_desc[seg_mid]
        seg_small_orig = sorted_idx_desc[seg_small]

        if increase_edges:

            if len(seg_small_orig)>0:
                S_new[seg_small_orig] *= ratio_clamped
        else:

            if len(seg_big_orig)>0:
                S_new[seg_big_orig] /= (ratio_clamped+1e-10)
        return S_new

    def structure_perturbation_spco(self, g, increase_edges=True):
        """
        SpCo 
        """
        try:
            # A_sp = g.adj(scipy_fmt='coo')
            # A = A_sp.toarray().astype(float)
            src, dst = g.edges()
            A = np.zeros((g.num_nodes(), g.num_nodes()), dtype=np.float32)
            np.add.at(A, (src.cpu().numpy(), dst.cpu().numpy()), 1)
        except Exception as e:
            warnings.warn(f"[WARN] dense: {e}")
            return g.clone()

        N = A.shape[0]

        deg = A.sum(axis=1)  # (N,)
        deg_inv_sqrt = np.zeros_like(deg)
        valid = (deg>0)
        deg_inv_sqrt[valid] = 1./np.sqrt(deg[valid])
        D_inv_sqrt = spdiags(deg_inv_sqrt, 0, N, N)  
        A_tilde = D_inv_sqrt @ A @ D_inv_sqrt
        L = eye(N) - A_tilde


        delta_add = np.ones((N,N))
        delta_del = np.ones((N,N))

        # 3) C = (1 - spco_theta)*L, or (pure L if theta=1)
        C = (1.0 - self.spco_theta)*L
        # print(type(C)) 
        # C = C.toarray()  
        C = np.array(C)

        # 4) 计算 K_add, K_del, 并做 sinkhorn
        alpha_add =  2.0 * np.sum(C * delta_add) / self.spco_epsilon
        alpha_del = -2.0 * np.sum(C * delta_del) / self.spco_epsilon

        K_add = np.exp(alpha_add * C)
        K_del = np.exp(alpha_del * C)


        dist = deg / (deg.sum() + 1e-10)

        delta_add_new = self._sinkhorn_normalize(K_add, dist, self.spco_sinkhorn_iter)
        delta_del_new = self._sinkhorn_normalize(K_del, dist, self.spco_sinkhorn_iter)

        Delta = delta_add_new - delta_del_new


        A_new = A + self.spco_lam * Delta

        A_new = 0.5*(A_new + A_new.T)
        A_new[A_new<0] = 0


        if not self.weighted_graph:
            if increase_edges:
                A_bin = (A_new > self.threshold).astype(float)
            else:
                A_bin = (A_new < self.threshold).astype(float)
        else:

            A_bin = np.clip(A_new, 0, 1)

        A_sp2 = csr_matrix(A_bin)
        src, dst = A_sp2.nonzero()
        g_new = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)

        if self.weighted_graph:
            g_new.edata['weight'] = torch.tensor(A_sp2.data, dtype=torch.float32)

        if 'attr' in g.ndata:
            g_new.ndata['attr'] = g.ndata['attr'].clone()

        return g_new

    def _sinkhorn_normalize(self, K, dist, n_iter=3):
        """

        """
        N = len(dist)
        dist_ = dist.reshape(-1,1)   # (N,1)

        K_ = K * (1./dist).reshape(1,-1)  

        u = np.ones((N,1)) / N
        for _ in range(n_iter):
            # v = dist_ / (K.T @ u)
            # u = 1.0 / (K_ @ v)

            v = dist_ / (K.T.dot(u)+1e-10)
            u = 1.0 / (K_.dot(v)+1e-10)


        delta = np.diag(u[:,0]) @ K @ np.diag(v[:,0])
        return delta
    