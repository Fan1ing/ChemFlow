import torch.nn as nn
import torch
from torch_geometric.nn import  Set2Set
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalConditionedMultiHeadAttention(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cond_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cond_dim = cond_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.use_cond = cond_dim > 0
        if self.use_cond:
            self.gq_mod_proj = nn.Linear(cond_dim, self.head_dim, bias=False)
            self.gk_mod_proj = nn.Linear(cond_dim, self.head_dim, bias=False)
            self.bias_alpha = nn.Parameter(torch.tensor(0.1))
            self.cross_bias_alpha = nn.Parameter(torch.tensor(0.01))

    def _shape(self, x, B, L):
        # [B, L, D] -> [B, H, L, Dh]
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _expand_cond(self, cond, L):
        """
        cond:
          - [B, C]
          - [B, L, C]
        return:
          - [B, L, C]
        """
        if cond is None:
            return None
        if cond.dim() == 2:
            return cond.unsqueeze(1).expand(-1, L, -1)
        if cond.dim() == 3:
            return cond
        raise ValueError(f"cond shape error: {cond.shape}")

    def _project_global(self, cond_q, cond_k, H):
        """
        cond_q: [B, Lq, C]
        cond_k: [B, Lk, C]
        return:
          gq: [B, H, Lq, Dh]
          gk: [B, H, Lk, Dh]
        """
        gq = self.gq_mod_proj(cond_q).unsqueeze(1).expand(-1, H, -1, -1)
        gk = self.gk_mod_proj(cond_k).unsqueeze(1).expand(-1, H, -1, -1)
        return gq, gk

    def _compute_global_modulation(self, qh, kh, cond_q=None, cond_k=None):
        """

        qh: [B, H, Lq, Dh]
        kh: [B, H, Lk, Dh]
        cond_q: [B, Lq, C] or [B, C]
        cond_k: [B, Lk, C] or [B, C]
        return: [B, H, Lq, Lk]
        """
        B, H, Lq, Dh = qh.shape
        _, _, Lk, _ = kh.shape

        cond_q = self._expand_cond(cond_q, Lq)
        cond_k = self._expand_cond(cond_k, Lk)

        if (cond_q is None) or (cond_k is None):
            return qh.new_zeros(B, H, Lq, Lk)

        gq, gk = self._project_global(cond_q, cond_k, H)

        # [B,H,Lq,Dh] x [B,H,Dh,Lk] -> [B,H,Lq,Lk]
        mod = torch.matmul(gq, gk.transpose(-2, -1)) / (Dh ** 0.5)
        return mod

    def _compute_global_local_cross_bias(self, qh, kh, cond_q=None, cond_k=None):
        """
          q_i^T g_j + g_i^T k_j

        qh: [B, H, Lq, Dh]
        kh: [B, H, Lk, Dh]
        cond_q: [B, Lq, C] or [B, C]
        cond_k: [B, Lk, C] or [B, C]
        return: [B, H, Lq, Lk]
        """
        B, H, Lq, Dh = qh.shape
        _, _, Lk, _ = kh.shape

        cond_q = self._expand_cond(cond_q, Lq)
        cond_k = self._expand_cond(cond_k, Lk)

        if (cond_q is None) or (cond_k is None):
            return qh.new_zeros(B, H, Lq, Lk)

        gq, gk = self._project_global(cond_q, cond_k, H)

        q_gk = torch.matmul(qh, gk.transpose(-2, -1)) / (Dh ** 0.5)  # [B,H,Lq,Lk]
        gq_k = torch.matmul(gq, kh.transpose(-2, -1)) / (Dh ** 0.5)  # [B,H,Lq,Lk]

        return gq_k

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,          # [B, Lq, Lk] or [B,1,Lq,Lk] or [B,H,Lq,Lk], True=valid
        key_padding_mask=None,   # [B, Lk], True=padding
        cond_q=None,             # [B, Lq, cond_dim] or [B, cond_dim]
        cond_k=None,             # [B, Lk, cond_dim] or [B, cond_dim]
        need_weights=True,
    ):
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        q = self._shape(self.q_proj(query), B, Lq)   # [B,H,Lq,Dh]
        k = self._shape(self.k_proj(key),   B, Lk)   # [B,H,Lk,Dh]
        v = self._shape(self.v_proj(value), B, Lk)   # [B,H,Lk,Dh]

        local_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Lq,Lk]

        if self.use_cond and (cond_q is not None or cond_k is not None):
            global_mod = self._compute_global_modulation(
                q, k, cond_q=cond_q, cond_k=cond_k
            )  # [B,H,Lq,Lk]
            logits = local_logits

            cross_bias = self._compute_global_local_cross_bias(
                q, k, cond_q=cond_q, cond_k=cond_k
            )  # [B,H,Lq,Lk]

        else:
            logits = local_logits

        # 掩码
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B,1,Lq,Lk]
            logits = logits.masked_fill(~attn_mask, float("-inf"))

        if key_padding_mask is not None:
            # True 表示 padding
            logits = logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,Lq,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(out)

        if need_weights:
            return out, attn
        return out, None


class FeatureCrossAttention(nn.Module):
    def __init__(self, dim_in_q, dim_in_kv, model_dim, num_heads, dropout=0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = 8

        self.q_map = nn.Linear(dim_in_q, 32)
        self.k_map = nn.Linear(dim_in_kv, 32)
        self.v_map = nn.Linear(dim_in_kv, 32)

        self.out_map = nn.Linear(32, model_dim)
        self.Qout = nn.Linear(32, model_dim)
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
        out = Qm_ + out

        return out, attn


class CrossMolGroupInter(nn.Module):


    def __init__(
        self,
        group_dim: int,
        K: int,
        mol_feat_dim: int,
        mol_id_emb_dim: int = 18,
        num_heads: int = 4,
        use_set2set: bool = True,
        s2s_steps: int = 2,
        bidirectional_cross: bool = True,
        cond_dim: int = 41,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.group_dim = group_dim
        self.bidirectional_cross = bidirectional_cross
        self.cond_dim = cond_dim

        self.group_content_dim = group_dim - cond_dim
        assert self.group_content_dim > 0

        self.in_dim = group_dim + mol_id_emb_dim

        # molecule id embedding
        self.mol_emb = nn.Linear(K, mol_id_emb_dim, bias=False)

        # group token -> attention input
        self.group_token_proj = nn.Linear(group_dim, self.in_dim)

        # molecule token -> attention input
        self.mol_proj = nn.Linear(mol_feat_dim, self.in_dim, bias=False)

        # self-attn on group tokens
        self.self_attn1 = GlobalConditionedMultiHeadAttention(
            embed_dim=self.in_dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        self.self_attn2 = GlobalConditionedMultiHeadAttention(
            embed_dim=self.in_dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        # group(query) <- mol(key/value)


        self.norm1 = nn.LayerNorm(self.in_dim)
        self.norm2 = nn.LayerNorm(self.in_dim)
        self.norm3 = nn.LayerNorm(self.in_dim)
        self.norm4 = nn.LayerNorm(self.in_dim)


        self.norm_ffn_group1 = nn.LayerNorm(self.in_dim)
        self.norm_ffn_group2 = nn.LayerNorm(self.in_dim)



        # === cross-attn: group(query) attend mol(key/value) ===
        self.mol_proj = nn.Linear(mol_feat_dim, self.in_dim, bias=False)
        self.cross_mha = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(self.in_dim)



        # === optional reverse cross-attn: mol(query) attend group(key/value) ===
        if self.bidirectional_cross:
            self.cross_mha_rev = nn.MultiheadAttention(self.in_dim, num_heads, batch_first=True)
            self.norm_cross_rev = nn.LayerNorm(self.in_dim)

            self.norm_ffn1 = nn.LayerNorm(self.in_dim)


        self.norm_ffn = nn.LayerNorm(self.in_dim)

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
            return int(max((int(gb.max()) if gb.numel() > 0 else -1) for gb in gb_list) + 1)
        return 1

    def _split_group_cond(self, xg):
        # xg: [N, group_dim] = [content ; cond]

        cond = xg[:, -self.cond_dim:]
        xg = xg[:, :self.group_dim]
        return xg, cond

    def forward(
        self,
        xg_list,
        gb_list,
        mol_tok_raw=None,       # [B_sub, K, mol_feat_dim]
        expanded_x_flat=None,   # [B_sub*K, mol_feat_dim]
        mol_key_pad_mask=None,
        return_attn=False,
    ):
        device = xg_list[0].device
        K = self.K
        B_sub = self._compute_B_sub(gb_list)

        # ===== 0) molecule tokens =====
        if mol_tok_raw is None:
            if expanded_x_flat is None:
                raise ValueError("必须提供 mol_tok_raw 或 expanded_x_flat")
            mol_tok_raw = expanded_x_flat.view(B_sub, K, -1)

        mol_tok = self.mol_proj(mol_tok_raw)  # [B_sub,K,in_dim]

        if mol_tok_raw.size(-1) >= self.cond_dim:
            mol_cond = mol_tok_raw[..., -self.cond_dim:]
        else:
            mol_cond = mol_tok_raw.new_zeros(B_sub, K, self.cond_dim)

        # ===== 1) flatten group tokens =====
        tokens_all, conds_all, token_b, token_bi, token_mol = [], [], [], [], []
        for i in range(K):
            xg_i, gb_i = xg_list[i], gb_list[i]
            if xg_i.numel() == 0:
                continue
            xg_full, cond_i = self._split_group_cond(xg_i)



            one_hot = F.one_hot(torch.tensor(i, device=device), num_classes=K).float().unsqueeze(0)
            me = self.mol_emb(one_hot).expand(xg_i.size(0), -1)

            token_i = xg_full
            token_i = torch.cat([xg_full, me], dim=1) if me.size(1) < token_i.size(1) else token_i


            tokens_all.append(token_i)
            conds_all.append(cond_i)
            token_b.append(gb_i)
            token_bi.append(gb_i * K + i)
            token_mol.append(torch.full((xg_i.size(0),), i, device=device, dtype=torch.long))

        if len(tokens_all) == 0:
            per_mol_out = [torch.zeros(B_sub, self.group_dim, device=device) for _ in range(K)]
            mix_feat = torch.zeros(B_sub, 2 * self.in_dim, device=device) if self.use_set2set else None
            if return_attn:
                return per_mol_out, mix_feat, {}
            return per_mol_out, mix_feat, mol_tok

        feats = torch.cat(tokens_all, dim=0)       # [N_tok, in_dim]
        conds = torch.cat(conds_all, dim=0)        # [N_tok, cond_dim]
        b_idx = torch.cat(token_b, dim=0).long()
        bi_idx = torch.cat(token_bi, dim=0).long()
        mol_id = torch.cat(token_mol, dim=0).long()

        # ===== 2) bucket by mixture and pad =====
        sort_order = torch.argsort(b_idx)
        feats_sorted = feats.index_select(0, sort_order)
        conds_sorted = conds.index_select(0, sort_order)
        b_sorted = b_idx.index_select(0, sort_order)
        bi_sorted = bi_idx.index_select(0, sort_order)
        mol_sorted = mol_id.index_select(0, sort_order)

        counts = torch.bincount(b_sorted, minlength=B_sub)
        feat_chunks = torch.split(feats_sorted, counts.tolist())
        cond_chunks = torch.split(conds_sorted, counts.tolist())

        padded = pad_sequence(feat_chunks, batch_first=True, padding_value=0.0)   # [B_sub,L_max,in_dim]
        padded_cond = pad_sequence(cond_chunks, batch_first=True, padding_value=0.0)  # [B_sub,L_max,cond_dim]

        L_max = padded.size(1)
        arange_L = torch.arange(L_max, device=device).unsqueeze(0)
        key_pad_mask = arange_L >= counts.unsqueeze(1)  # [B_sub,L_max], True=padding

        origin = padded
        padded = self.norm1(origin)
        # ===== 3) self-attn 1 =====
        attn_out1, attn_mat1 = self.self_attn1(
            padded, padded, padded,
            key_padding_mask=key_pad_mask,
            cond_q=padded_cond,
            cond_k=padded_cond,
            need_weights=True,
        )
        x = origin+attn_out1
        x1 = self.norm2(x)
        # ===== 4) self-attn 2 =====
        attn_out2, attn_mat2 = self.self_attn2(
            x1, x1, x1,
            key_padding_mask=key_pad_mask,
            cond_q=padded_cond,
            cond_k=padded_cond,
            need_weights=True,
        )
        x = x+attn_out2
        # ========== 4) cross-attn：group(query) <- mol(key/value) ==========
        x2 = self.norm_cross(x)
        mol_tok1 = self.norm3(mol_tok)
        cross_out, cross_w = self.cross_mha(
            query=x2,
            key=mol_tok1,
            value=mol_tok1,
            key_padding_mask=mol_key_pad_mask,  # None 或 [B_sub,K]
            need_weights=return_attn
        )
        x = cross_out+x

        # ========== 6) cross-attn mol(query) <- group(key/value) ==========
        if self.bidirectional_cross:
            mol_tok2 = self.norm_cross_rev(mol_tok)
            x1 = self.norm4(x)
            mol_out, mol_w = self.cross_mha_rev(
                query=mol_tok2,
                key=x1,
                value=x1,
                key_padding_mask=key_pad_mask,
                need_weights=return_attn
            )

            mol_tok = mol_tok + mol_out


        else:
            mol_w = None


        # ===== 7) remove pad =====
        valid_mask = (torch.arange(L_max, device=device)[None, :] < counts[:, None])
        x_flat = x.reshape(-1, x.size(-1))[valid_mask.view(-1)]

        N = feats.size(0)
        inv = torch.empty_like(sort_order)
        inv[sort_order] = torch.arange(N, device=device)
        attn_unsorted = x_flat[inv]

        # ===== 8) readout =====
        mol_id_per_token = (bi_idx % self.K)

        per_mol_out = []
        for i in range(self.K):
            mask_i = (mol_id_per_token == i)
            if mask_i.any():
                part_i = attn_unsorted[mask_i]
                b_idx_i = b_idx[mask_i]
                if self.use_set2set:
                    s2s_i = self.mol_s2s(part_i, b_idx_i)
                else:
                    s2s_i = part_i.new_zeros(B_sub, 2 * self.in_dim)
                    for b in range(B_sub):
                        mb = (b_idx_i == b)
                        if mb.any():
                            m = part_i[mb].mean(dim=0)
                            s2s_i[b, :self.in_dim] = m
                            s2s_i[b, self.in_dim:] = m
            else:
                s2s_i = attn_unsorted.new_zeros(B_sub, 2 * self.in_dim)

            per_mol_out.append(self.readout(s2s_i))

        if self.use_set2set:
            mix_feat = self.mix_s2s(attn_unsorted, b_idx)
        else:
            mix_feat = None

        if return_attn:
            b_offsets = torch.zeros(B_sub + 1, dtype=torch.long, device=device)
            b_offsets[1:] = torch.cumsum(counts, dim=0)
            info = {
                "attn_self_1": attn_mat1,
                "attn_self_2": attn_mat2,
                "attn_cross_g2m": cross_w,
                "attn_cross_m2g": mol_w,
                "lengths": counts,
                "b_offsets": b_offsets,
                "mol_sorted": mol_sorted,
                "bi_sorted": bi_sorted,
            }
            return per_mol_out, mix_feat, info

        return per_mol_out, mix_feat, mol_tok
