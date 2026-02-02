import torch.nn as nn
import torch
from torch_geometric.nn import  Set2Set
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence





class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = 32

        self.q_map = nn.Linear(dim_in_q, 128)
        self.k_map = nn.Linear(dim_in_kv, 128)
        self.v_map = nn.Linear(dim_in_kv, 128)

        self.out_map = nn.Linear(128, model_dim)
        self.Qout = nn.Linear(128, model_dim)
        self.norm = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale   = 1 / math.sqrt(self.d_k)

    def forward(self, Q_in, KV_in, mask=None):
        """
        Q_in:  (B, L_q, dim_in_q)
        KV_in: (B, L_kv, dim_in_kv)
        """


        # ========== 1) Mapping ==========
        Qm = self.q_map(Q_in)  # (B, L_q, model_dim)
        Km = self.k_map(KV_in) # (B, L_kv, model_dim)
        Vm = self.v_map(KV_in) # (B, L_kv, model_dim)
        B, L_q, Dq = Qm.shape
        _, L_kv, Dk = Km.shape
        # ========== 2) multiple head ==========
        Qh = Qm.view(B, L_q, self.num_heads, self.d_k).permute(0, 2, 3, 1)  # (B, H, d_k, L_q)
        Kh = Km.view(B, L_kv, self.num_heads, self.d_k).permute(0, 2, 3, 1) # (B, H, d_k, L_kv)
        Vh = Vm.view(B, L_kv, self.num_heads, self.d_k).permute(0, 2, 3, 1) # (B, H, d_k, L_kv)

        # ========== 3) Exchange feature dimension & token dimension ==========
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # (B, H, d_k, d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out_h = torch.matmul(attn, Vh)  # (B, H, d_k, L_q)
        out_h = out_h.permute(0, 3, 1, 2).contiguous().view(B, L_q, self.num_heads * self.d_k)  # (B, L_q, model_dim)
        out = self.out_map(out_h)
        Qm_ = self.Qout(Qm)
        out = self.norm(Qm_ + out)

        return out, attn


'''class CrossMolGroupInter(nn.Module):

    def __init__(self, group_dim: int, K: int, mol_emb_dim: int = 18,
                 num_heads: int = 4, use_set2set: bool = True, s2s_steps: int = 2):
        super().__init__()
        self.K = K
        self.group_dim = group_dim
        self.in_dim = group_dim + mol_emb_dim
        self.mol_emb = nn.Linear(K, mol_emb_dim, bias=False)

        self.mha = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)


        # === 2(FFN) ===
        self.ffn = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim),
            nn.ReLU(),
            nn.Linear(2 * self.in_dim, self.in_dim),
        )
        self.norm2 = nn.LayerNorm(self.in_dim)
        self.norm3 = nn.LayerNorm(self.in_dim)

        # === 3. readout ===
        self.readout = nn.Sequential(
            nn.Linear(self.in_dim * 2, group_dim),
            nn.ReLU(),
            nn.Linear(group_dim, group_dim),
        )



        self.use_set2set = use_set2set
        if use_set2set:
            self.mol_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # moledule level
            self.mix_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # mix level

    def forward(self, xg_list, gb_list, return_attn=False):
        """
        xg_list: [xg1, xg2, xg3], xg_i: [Gi, group_dim]
        gb_list: [gb1, gb2, gb3], gb_i: [Gi] in [0..B_sub-1]
        """
        device = xg_list[0].device
        K = self.K

        # Calculate the number of mixtures in the same mini batch (B-sub)
        if any(gb.numel() > 0 for gb in gb_list):
            B_sub = int(max((int(gb.max()) if gb.numel() > 0 else -1) for gb in gb_list) + 1)
        else:
            B_sub = 1

        # ==== 1) Splicing all tokens (with molecular ID embedding) ====
        tokens_all, token_b, token_bi, token_mol = [], [], [], []
        for i in range(K):
            xg_i, gb_i = xg_list[i], gb_list[i]
            if xg_i.numel() == 0:
                continue

            #   one-hot
            one_hot = F.one_hot(torch.tensor(i, device=device), num_classes=K).float()  # [K]
            one_hot = one_hot.unsqueeze(0)  # [1, K]
            me = self.mol_emb(one_hot)      # [1, mol_emb_dim]
            me = me.expand(xg_i.size(0), -1)  # [Gi, mol_emb_dim]

            t = torch.cat([xg_i, me], dim=1)                                    # [Gi, H+Em]
            tokens_all.append(t)
            token_b.append(gb_i)
            token_bi.append(gb_i * K + i)                                       # [Gi]
            token_mol.append(torch.full((xg_i.size(0),), i, device=device, dtype=torch.long))

        if len(tokens_all) == 0:
            per_mol_out = [torch.zeros(B_sub, self.group_dim, device=device) for _ in range(K)]
            mix_feat = torch.zeros(B_sub, 2 * self.in_dim, device=device) if self.use_set2set else None
            return per_mol_out, mix_feat

        feats   = torch.cat(tokens_all, dim=0)         # [N_tok, H_in]
        b_idx   = torch.cat(token_b,   dim=0).long()   # [N_tok]  mixture id
        bi_idx  = torch.cat(token_bi,  dim=0).long()   # [N_tok]  global (b,i) id
        mol_id = torch.cat(token_mol, dim=0).long()  # [N_tok]

        # ==== 2) Construct a 'batch sequence' grouped by mixture ====
        sort_order = torch.argsort(b_idx)              # [N_tok]
        feats_sorted  = feats.index_select(0, sort_order)
        b_sorted      = b_idx.index_select(0, sort_order)
        bi_sorted     = bi_idx.index_select(0, sort_order)
        mol_sorted = mol_id.index_select(0, sort_order)  # [N_tok]

        # How many tokens does each B have：
        counts = torch.bincount(b_sorted, minlength=B_sub)  # [B_sub]
        # Cut b into a list
        chunks = torch.split(feats_sorted, counts.tolist())
        # pad
        padded = pad_sequence(chunks, batch_first=True, padding_value=0.0)      # [B_sub, L_max, H_in]

        L_max = padded.size(1)
        lens = counts
        arange_L = torch.arange(L_max, device=device).unsqueeze(0)              # [1, L_max]
        key_pad_mask = arange_L >= lens.unsqueeze(1)                            # [B_sub, L_max], bool

        attn_out, attn_mat1 = self.mha(padded, padded, padded, key_padding_mask=key_pad_mask)  # [B_sub, L_max, H_in]

        padded = self.norm2(padded + attn_out)
        attn_out, attn_mat2 = self.mha2(padded, padded, padded, key_padding_mask=key_pad_mask)
        x = self.norm3(padded + attn_out)

        # === 4) delet pad ===
        valid_mask = (torch.arange(L_max, device=device)[None, :] < lens[:, None])
        x_flat = x.reshape(-1, x.size(-1))[valid_mask.view(-1)]

        N = feats.size(0)
        inv = torch.empty_like(sort_order)
        inv[sort_order] = torch.arange(N, device=device)
        attn_unsorted = x_flat[inv]  # [N_tok, H_in]                # [N_tok, H_in]             # [N_tok, H_in]

        # ==== 4) readout ====
        mol_id_per_token = (bi_idx % self.K)  # [N_tok]



        per_mol_out = []
        for i in range(self.K):
            mask_i = (mol_id_per_token == i)
            if mask_i.any():
                part_i = attn_unsorted[mask_i]  # [N_i, H_in]
                b_idx_i = b_idx[mask_i]  # [N_i]
                s2s_i = self.mol_s2s(part_i, b_idx_i)  # [B_sub, 2*H_in]
            else:
                s2s_i = attn_unsorted.new_zeros(B_sub, 2 * self.in_dim)
            per_mol_out.append(self.readout(s2s_i))  # [B_sub, group_dim]

        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)          # [B_sub, 2*H_in]
        else:
            mix_feat = None

        if return_attn:
            b_offsets = torch.zeros(B_sub + 1, dtype=torch.long, device=device)
            b_offsets[1:] = torch.cumsum(counts, dim=0)

            return per_mol_out, mix_feat, {
                "attn1": attn_mat1,               # [B_sub, h, L_max, L_max]
                "attn2": attn_mat2,               # [B_sub, h, L_max, L_max]
                "lengths": lens,                  # [B_sub]
                "counts": counts,                 # [B_sub]
                "b_offsets": b_offsets,           # [B_sub+1]
                "mol_sorted": mol_sorted,         # [N_tok_sorted]
            }

        return per_mol_out, mix_feat'''




class CrossMolGroupInter(nn.Module):
    """
    Group tokens (from xg_list) self-attn + cross-attn with molecule-level tokens (from expanded_x)
    """

    def __init__(
        self,
        group_dim: int,
        K: int,
        mol_feat_dim: int,            # == expanded_x最后一维（拼完global_node_attr后的维度）
        mol_id_emb_dim: int = 18,     # 你原来叫 mol_emb_dim
        num_heads: int = 4,
        use_set2set: bool = True,
        s2s_steps: int = 2,
        bidirectional_cross: bool = True,  # True: 再做一次 mol->group 反向cross
    ):
        super().__init__()
        self.K = K
        self.group_dim = group_dim
        self.bidirectional_cross = bidirectional_cross

        # group token 维度： group_dim + 分子ID嵌入(mol_id_emb_dim)
        self.in_dim = group_dim + mol_id_emb_dim

        # 分子ID one-hot -> emb
        self.mol_emb = nn.Linear(K, mol_id_emb_dim, bias=False)

        # === group self-attn (两层) ===
        self.mha = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(self.in_dim)
        self.norm3 = nn.LayerNorm(self.in_dim)


        self.ffn_group1 = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim),
            nn.ReLU(),
            nn.Linear(2 * self.in_dim, self.in_dim),
        )
        self.norm_ffn_group1 = nn.LayerNorm(self.in_dim)

        self.ffn_group2 = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim),
            nn.ReLU(),
            nn.Linear(2 * self.in_dim, self.in_dim),
        )
        self.norm_ffn_group2 = nn.LayerNorm(self.in_dim)



        # === cross-attn: group(query) attend mol(key/value) ===
        self.mol_proj = nn.Linear(mol_feat_dim, self.in_dim, bias=False)
        self.cross_mha = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(self.in_dim)



        # === optional reverse cross-attn: mol(query) attend group(key/value) ===
        if self.bidirectional_cross:
            self.cross_mha_rev = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
            self.norm_cross_rev = nn.LayerNorm(self.in_dim)
            self.ffn1 = nn.Sequential(
                nn.Linear(self.in_dim, 2 * self.in_dim),
                nn.ReLU(),
                nn.Linear(2 * self.in_dim, self.in_dim),
            )
            self.norm_ffn1 = nn.LayerNorm(self.in_dim)

        # === FFN（可选增强，这里保留但不强依赖）===
        self.ffn = nn.Sequential(
            nn.Linear(self.in_dim, 2 * self.in_dim),
            nn.ReLU(),
            nn.Linear(2 * self.in_dim, self.in_dim),
        )
        self.norm_ffn = nn.LayerNorm(self.in_dim)

        # === readout：你原来的逻辑 ===
        self.readout = nn.Sequential(
            nn.Linear(self.in_dim * 2, group_dim),
            nn.ReLU(),
            nn.Linear(group_dim, group_dim),
        )

        self.use_set2set = use_set2set
        if use_set2set:
            self.mol_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # per mol
            self.mix_s2s = Set2Set(self.in_dim, processing_steps=s2s_steps)  # mixture

    @staticmethod
    def _compute_B_sub(gb_list):
        if any(gb.numel() > 0 for gb in gb_list):
            B_sub = int(max((int(gb.max()) if gb.numel() > 0 else -1) for gb in gb_list) + 1)
        else:
            B_sub = 1
        return B_sub

    def forward(
        self,
        xg_list,                # list length K, each xg_i: [Gi, group_dim]
        gb_list,                # list length K, each gb_i: [Gi] mixture id in [0..B_sub-1]
        mol_tok_raw=None,       # [B_sub, K, mol_feat_dim]  (推荐)
        expanded_x_flat=None,   # [B_sub*K, mol_feat_dim]   (可选替代输入)
        mol_key_pad_mask=None,  # [B_sub, K] bool, True=padding (可选)
        return_attn=False,
    ):

        device = xg_list[0].device
        K = self.K
        B_sub = self._compute_B_sub(gb_list)

        # ========== 0) mol tokens ==========
        if mol_tok_raw is None:
            if expanded_x_flat is None:
                raise ValueError("必须提供 mol_tok_raw=[B_sub,K,mol_feat_dim] 或 expanded_x_flat=[B_sub*K,mol_feat_dim]")
            if expanded_x_flat.dim() != 2 or expanded_x_flat.size(0) != B_sub * K:
                raise ValueError(f"expanded_x_flat 形状应为 [B_sub*K, mol_feat_dim]，但拿到 {tuple(expanded_x_flat.shape)}")
            mol_tok_raw = expanded_x_flat.view(B_sub, K, -1)  # [B_sub, K, mol_feat_dim]

        if mol_tok_raw.dim() != 3 or mol_tok_raw.size(0) != B_sub or mol_tok_raw.size(1) != K:
            raise ValueError(f"mol_tok_raw 形状应为 [B_sub,K,mol_feat_dim]，但拿到 {tuple(mol_tok_raw.shape)}")

        mol_tok = self.mol_proj(mol_tok_raw)  # [B_sub, K, in_dim]

        # ========== 1) 拼接组团 tokens（带mol id embedding）==========
        tokens_all, token_b, token_bi, token_mol = [], [], [], []
        for i in range(K):
            xg_i, gb_i = xg_list[i], gb_list[i]
            if xg_i.numel() == 0:
                continue

            one_hot = F.one_hot(torch.tensor(i, device=device), num_classes=K).float().unsqueeze(0)  # [1,K]
            me = self.mol_emb(one_hot).expand(xg_i.size(0), -1)  # [Gi, mol_id_emb_dim]
            t = torch.cat([xg_i, me], dim=1)  # [Gi, in_dim]

            tokens_all.append(t)
            token_b.append(gb_i)
            token_bi.append(gb_i * K + i)  # [Gi]
            token_mol.append(torch.full((xg_i.size(0),), i, device=device, dtype=torch.long))

        if len(tokens_all) == 0:
            per_mol_out = [torch.zeros(B_sub, self.group_dim, device=device) for _ in range(K)]
            mix_feat = torch.zeros(B_sub, 2 * self.in_dim, device=device) if self.use_set2set else None
            if return_attn:
                return per_mol_out, mix_feat, {}
            return per_mol_out, mix_feat

        feats = torch.cat(tokens_all, dim=0)          # [N_tok, in_dim]
        b_idx = torch.cat(token_b, dim=0).long()      # [N_tok]
        bi_idx = torch.cat(token_bi, dim=0).long()    # [N_tok]
        mol_id = torch.cat(token_mol, dim=0).long()   # [N_tok]

        # ========== 2) mixture分桶 -> pad ==========
        sort_order = torch.argsort(b_idx)  # [N_tok]
        feats_sorted = feats.index_select(0, sort_order)
        b_sorted = b_idx.index_select(0, sort_order)
        bi_sorted = bi_idx.index_select(0, sort_order)
        mol_sorted = mol_id.index_select(0, sort_order)

        counts = torch.bincount(b_sorted, minlength=B_sub)  # [B_sub]
        chunks = torch.split(feats_sorted, counts.tolist())
        padded = pad_sequence(chunks, batch_first=True, padding_value=0.0)  # [B_sub, L_max, in_dim]

        L_max = padded.size(1)
        lens = counts
        arange_L = torch.arange(L_max, device=device).unsqueeze(0)
        key_pad_mask = arange_L >= lens.unsqueeze(1)  # [B_sub, L_max]

        # ========== 3) group self-attn（两层）==========
        attn_out1, attn_mat1 = self.mha(padded, padded, padded, key_padding_mask=key_pad_mask)
        x = self.norm2(padded + attn_out1)
        f = self.ffn_group1(x)
        x = self.norm_ffn_group1(x + f)

        attn_out2, attn_mat2 = self.mha2(x, x, x, key_padding_mask=key_pad_mask)
        x = self.norm3(x + attn_out2)
        f = self.ffn_group2(x)
        x = self.norm_ffn_group2(x + f)

        # ========== 4) cross-attn：group(query) <- mol(key/value) ==========
        cross_out, cross_w = self.cross_mha(
            query=x,
            key=mol_tok,
            value=mol_tok,
            key_padding_mask=mol_key_pad_mask,  # None 或 [B_sub,K]
            need_weights=return_attn
        )
        x = self.norm_cross(x + cross_out)
        x1 =x
        # ========== 5) 可选：再过一层FFN ==========
        f = self.ffn(x)
        x = self.norm_ffn(x + f)

        # ========== 6) 可选：反向cross-attn mol(query) <- group(key/value) ==========
        if self.bidirectional_cross:
            mol_out, mol_w = self.cross_mha_rev(
                query=mol_tok,
                key=x1,
                value=x1,
                key_padding_mask=key_pad_mask,
                need_weights=return_attn
            )
            mol_tok = self.norm_cross_rev(mol_tok + mol_out)
            f = self.ffn1(mol_tok)
            mol_tok = self.norm_ffn1(mol_tok + f)



        else:
            mol_w = None

        # ========== 7) 删除pad，回到 token 维度 ==========
        valid_mask = (torch.arange(L_max, device=device)[None, :] < lens[:, None])  # [B_sub,L_max]
        x_flat = x.reshape(-1, x.size(-1))[valid_mask.view(-1)]  # [N_tok, in_dim]（按b_sorted顺序）

        N = feats.size(0)
        inv = torch.empty_like(sort_order)
        inv[sort_order] = torch.arange(N, device=device)
        attn_unsorted = x_flat[inv]  # [N_tok, in_dim]（回到原拼接顺序）

        # ========== 8) readout：按 mol_id_per_token 分K个输出 ==========
        mol_id_per_token = (bi_idx % self.K)  # [N_tok]

        per_mol_out = []
        for i in range(self.K):
            mask_i = (mol_id_per_token == i)
            if mask_i.any():
                part_i = attn_unsorted[mask_i]   # [N_i, in_dim]
                b_idx_i = b_idx[mask_i]          # [N_i]
                if self.use_set2set:
                    s2s_i = self.mol_s2s(part_i, b_idx_i)  # [B_sub, 2*in_dim]
                else:
                    # 如果不用Set2Set，就简单mean-pool（可按需改）
                    s2s_i = part_i.new_zeros(B_sub, 2 * self.in_dim)
                    for b in range(B_sub):
                        mb = (b_idx_i == b)
                        if mb.any():
                            m = part_i[mb].mean(dim=0)
                            s2s_i[b, :self.in_dim] = m
                            s2s_i[b, self.in_dim:] = m
            else:
                s2s_i = attn_unsorted.new_zeros(B_sub, 2 * self.in_dim)

            per_mol_out.append(self.readout(s2s_i))  # [B_sub, group_dim]

        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)  # [B_sub, 2*in_dim]
        else:
            mix_feat = None

        if return_attn:
            b_offsets = torch.zeros(B_sub + 1, dtype=torch.long, device=device)
            b_offsets[1:] = torch.cumsum(counts, dim=0)

            out_dict = {
                "attn_self_1": attn_mat1,   # [B_sub, num_heads, L_max, L_max]（pytorch版本可能是 [B_sub,L,L]）
                "attn_self_2": attn_mat2,
                "attn_cross_g2m": cross_w,  # group->mol
                "attn_cross_m2g": mol_w,    # mol->group (if bidirectional_cross)
                "lengths": lens,
                "counts": counts,
                "b_offsets": b_offsets,
                "mol_sorted": mol_sorted,
                "bi_sorted": bi_sorted,
            }
            return per_mol_out, mix_feat, out_dict

        return per_mol_out, mix_feat,mol_tok
