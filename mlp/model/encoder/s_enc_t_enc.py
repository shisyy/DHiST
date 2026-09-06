import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat
from mlp.model.blocks import GCN, WeightedPool
from mlp.model.encoder.t_enc_only import TEncoder
from mlp.model.utils import Graph

from mlp.model.encoder.unimatcher import CommonSequenceMatcher, LabelPriorSequenceMatcher

class FactorisedSTEncoder(nn.Module):
    def __init__(self, in_dim: int,
                    dim: int, 
                    max_snippet_len: int,
                    pooling: str,
                    dataset: str,
                    adaptive: str,
                    num_heads = 4, 
                    num_layers = 4,
                    gcn_layers = 5, 
                    drop_rate = 0.1,
                    shared = False, 
                    start_label_ratio = 1.0,
                    label_anneal_fraction = 0.125,
                    qahs_gate_floor = 0.0,
                    **kwargs):
        """
        S-Enc (Graph Convolutional Network & Attentive Pooling) + T-Enc (Self-Guided Parallel Attention Module)

        Args:
            in_dim (int): Model input dimensions (denoted as $d_{p}$ in our paper) 
            dim (int): Model hidden dimensions (denoted as $d$ in our paper) 
            max_snippet_len (int): max interval snippet number (denoted as $S$ in our paper) 
            pooling (str): Ways of pooling spatial features, including 'cls_token','weighted_pool','mean'
            dataset (str): _description_
            adaptive (str): Graph Convolutional Adjacency Matrix Initialization Approach
            num_heads (int, optional): The attention head of SGPA. Defaults to 4.
            num_layers (int, optional): The block numbers of SGPA. Defaults to 4.
            gcn_layers (int, optional): Defaults to 5.
            drop_rate (float, optional): Defaults to 0.1.
            shared (bool, optional): Represents whether T-Enc parameters are shared. Defaults to False.
        """
        super(FactorisedSTEncoder, self).__init__()

        self.shared = shared
        self.start_label_ratio = float(start_label_ratio)
        self.label_anneal_fraction = float(label_anneal_fraction)
        self.qahs_gate_floor = float(qahs_gate_floor)
        if not 0.0 <= self.qahs_gate_floor <= 1.0:
            raise ValueError("qahs_gate_floor must be within [0, 1]")
        self.spatial_blocks = DividedGraphConvolutionBlock(in_dim=in_dim, 
                                                           embed_dims=dim,
                                                           gcn_layers=gcn_layers, 
                                                           num_heads=num_heads, 
                                                           drop_rate=drop_rate, 
                                                           dataset=dataset, 
                                                           pooling=pooling,
                                                           adaptive=adaptive)
        self.temporal_feat_enc = TEncoder(dim=dim,
                                             max_snippet_len=max_snippet_len,
                                             num_heads=num_heads,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate,
                                             shared=shared)

        self.temporal_feat_enc_s = TEncoder(dim=dim,
                                             max_snippet_len=max_snippet_len,
                                             num_heads=num_heads,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate,
                                             shared=shared)
        self.temporal_feat_enc_e = TEncoder(dim=dim,
                                             max_snippet_len=max_snippet_len,
                                             num_heads=num_heads,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate,
                                             shared=shared)
        

        self.temporal_conv1d_1_h = MaskedConv1D(dim, dim, kernel_size=3, padding=1)
        self.temporal_conv1d_1_a_h = nn.ReLU()
        self.temporal_conv1d_2_h = MaskedConv1D(dim, 1, kernel_size=1)
        self.sigmoid_h = nn.Sigmoid()

        self.temporal_conv1d_1_s = MaskedConv1D(dim, dim, kernel_size=3, padding=1)
        self.temporal_conv1d_1_a_s = nn.ReLU()
        self.temporal_conv1d_2_s = MaskedConv1D(dim, 1, kernel_size=1)
        self.sigmoid_s = nn.Sigmoid()

        self.temporal_conv1d_1_e = MaskedConv1D(dim, dim, kernel_size=3, padding=1)
        self.temporal_conv1d_1_a_e = nn.ReLU()
        self.temporal_conv1d_2_e = MaskedConv1D(dim, 1, kernel_size=1)
        self.sigmoid_e = nn.Sigmoid()

    # def forward(self, mfeats, qfeats, m_mask, q_mask, h_mask=None, cur_step, total_step, idxs, mode, *args):
    def forward(self, mfeats, qfeats, m_mask, q_mask, cur_step, total_step, h_mask=None, uni_h_mask=None, uni_s_mask=None, uni_e_mask=None, mode=None):
        assert len(mfeats.shape) == 4
        # S-Enc+T-Enc = Factorized space-time encoder
        #m_mask: B, T
        mfeats = self.spatial_blocks(mfeats, m_mask) #B ,T ,C

        # hse_mask = torch.cat([uni_h_mask.unsqueeze(1), uni_s_mask.unsqueeze(1), uni_e_mask.unsqueeze(1)], dim=1)
        # hse_mask, _ = self.temporal_conv1d_1(hse_mask, m_mask.unsqueeze(1))
        # hse_mask = self.temporal_conv1d_1_a(hse_mask)
        # hse_mask, _ = self.temporal_conv1d_2(hse_mask, m_mask.unsqueeze(1))

        uni_h_score, _ = self.temporal_conv1d_1_h(mfeats.permute(0, 2, 1), m_mask.unsqueeze(1))
        uni_h_score = self.temporal_conv1d_1_a_h(uni_h_score)
        uni_h_score, _ = self.temporal_conv1d_2_h(uni_h_score, m_mask.unsqueeze(1)) #B, 1, T
        uni_h_score = self.sigmoid_h(uni_h_score)
        uni_h_score = uni_h_score.permute(0, 2, 1) #B, T, 1

        uni_s_score, _ = self.temporal_conv1d_1_s(mfeats.permute(0, 2, 1), m_mask.unsqueeze(1))
        uni_s_score = self.temporal_conv1d_1_a_s(uni_s_score)
        uni_s_score, _ = self.temporal_conv1d_2_s(uni_s_score, m_mask.unsqueeze(1)) #B, 1, T
        uni_s_score = self.sigmoid_s(uni_s_score)
        uni_s_score = uni_s_score.permute(0, 2, 1) #B, T, 1

        uni_e_score, _ = self.temporal_conv1d_1_e(mfeats.permute(0, 2, 1), m_mask.unsqueeze(1))
        uni_e_score = self.temporal_conv1d_1_a_e(uni_e_score)
        uni_e_score, _ = self.temporal_conv1d_2_e(uni_e_score, m_mask.unsqueeze(1)) #B, 1, T
        uni_e_score = self.sigmoid_e(uni_e_score)
        uni_e_score = uni_e_score.permute(0, 2, 1) #B, T, 1


        if mode != "Train":
            fusion_uni_h_mask = uni_h_score
            fusion_uni_s_mask = uni_s_score
            fusion_uni_e_mask = uni_e_score
        else:
            if uni_h_mask is None or uni_s_mask is None or uni_e_mask is None:
                raise ValueError("QAHS label masks are required during training")
            anneal_steps = max(1, int(total_step * self.label_anneal_fraction))
            progress = min(max(float(cur_step) / anneal_steps, 0.0), 1.0)
            used_label_ratio = self.start_label_ratio * (1.0 - progress)
            fusion_uni_h_mask = (
                uni_h_mask.unsqueeze(2) * used_label_ratio
                + uni_h_score * (1.0 - used_label_ratio)
            )
            fusion_uni_s_mask = (
                uni_s_mask.unsqueeze(2) * used_label_ratio
                + uni_s_score * (1.0 - used_label_ratio)
            )
            fusion_uni_e_mask = (
                uni_e_mask.unsqueeze(2) * used_label_ratio
                + uni_e_score * (1.0 - used_label_ratio)
            )


        fusion_uni_h_mask = self.qahs_gate_floor + (1.0 - self.qahs_gate_floor) * fusion_uni_h_mask
        fusion_uni_s_mask = self.qahs_gate_floor + (1.0 - self.qahs_gate_floor) * fusion_uni_s_mask
        fusion_uni_e_mask = self.qahs_gate_floor + (1.0 - self.qahs_gate_floor) * fusion_uni_e_mask

        mfeats_h = mfeats * fusion_uni_h_mask
        mfeats_h = mfeats_h.unsqueeze(2)
        ##need B, T, 1, C
        mfeats_h, qfeats_h = self.temporal_feat_enc(mfeats_h, qfeats, m_mask, q_mask, h_mask)

        mfeats_s = mfeats * fusion_uni_s_mask
        mfeats_s = mfeats_s.unsqueeze(2)
        ##need B, T, 1, C
        mfeats_s, qfeats_s = self.temporal_feat_enc_s(mfeats_s, qfeats, m_mask, q_mask, h_mask)

        mfeats_e = mfeats * fusion_uni_e_mask
        mfeats_e = mfeats_e.unsqueeze(2)
        ##need B, T, 1, C
        mfeats_e, qfeats_e = self.temporal_feat_enc_e(mfeats_e, qfeats, m_mask, q_mask, h_mask)

        mfeats = (mfeats_h + mfeats_s + mfeats_e) / 3
        qfeats = (qfeats_h + qfeats_s + qfeats_e) / 3

        return mfeats, qfeats, (uni_h_score.squeeze(2), uni_s_score.squeeze(2), uni_e_score.squeeze(2))

class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
    ):
        super(MaskedConv1D, self).__init__()

        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        assert x.size(-1) % self.stride == 0

        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)

        mask_float = mask.to(x.dtype)
        x = self.conv(x * mask_float)

        if self.stride > 1:
            mask_float = F.interpolate(
                mask_float, size=x.size(-1), mode='nearest'
            )
            mask = mask_float.bool()
        return x, mask

class MultiModalFusion(nn.Module):
    def __init__(self, n_feature, embed_dim=512, dropout=0.5):
        super().__init__()
        self.channel_avg = nn.AdaptiveAvgPool1d(1)  # 时序池化

        # 全局投影
        self.q_proj = nn.Linear(n_feature, embed_dim)
        self.k_proj = nn.Linear(n_feature, embed_dim)

        # 局部分支
        self.local_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

    def forward(self, x, temporal_mask=None):
        """
        x: (B, T, N, C)
        temporal_mask: (B, T)  # 1=有效帧, 0=无效帧
        """
        B, T, N, C = x.shape

        # ===== Step 1: 全局池化 (模态级全局语义) =====
        x_perm = x.permute(0, 2, 3, 1)                # (B, N, C, T)
        x_reshape = x_perm.reshape(B * N, C, T)       # (B*N, C, T)
        g = self.channel_avg(x_reshape).squeeze(-1)   # (B*N, C)
        g = g.reshape(B, N, C)                        # (B, N, C)

        # ===== Step 2: 模态级注意力 =====
        Q = self.q_proj(g)                            # (B, N, d)
        K = self.k_proj(g)                            # (B, N, d)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1)**0.5)
        attn = torch.softmax(attn, dim=-1)            # (B, N, N)

        # ===== Step 3: 局部分支 (bit-wise conv) =====
        x_local = self.local_conv(x_reshape)          # (B*N, d, T)
        x_local = x_local.reshape(B, N, -1, T).permute(0, 1, 3, 2)  # (B, N, T, d)

        # ===== Step 4: temporal mask (可选) =====
        if temporal_mask is not None:
            temporal_mask = temporal_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
            x_local = x_local * temporal_mask

        # ===== Step 5: 融合 (Softmax + Sigmoid 门控) =====
        fused = torch.einsum("bij,bjtd->bitd", attn, x_local)  # (B, N, T, d)
        fused = fused.permute(0,2,1,3)

        fused = torch.sigmoid(fused) * x                      # 门控调制 (⊗)

        # ===== Step 6: 残差连接 =====
        out = fused + x

        # mask 输出，防止 padding 污染
        # if temporal_mask is not None:
        #     out = out * temporal_mask

        return out



class DividedGraphConvolutionBlock(nn.Module):
    """Spatial Graph Convolution in Divided Space Time Attention.
        A warp for GCB+Attentive Pooling.
        
    Args:
        in_dim (int): The dimension of motion input
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        drop_rate (float): A Dropout layer on attn_output_weights. Defaults to
            0..
        pooling (str): The way to aggregate the spatial dimensions at the end.
        adaptive (str): How to apply adjacency matrix
    """

    def __init__(self, in_dim, embed_dims, gcn_layers=0, drop_rate=0, pooling='mean', dataset="babel", adaptive="none",
                 **kwargs):
        super().__init__()
        assert pooling in ['cls_token','weighted_pool','mean','max','sum']
        assert gcn_layers>0 and gcn_layers % 2 == 1
        self.pooling = pooling
        num_stage = (gcn_layers-1)//2
        
        # Generate adjacency matrix
        graph_cfg=dict(layout=dataset, mode='spatial', num_filter=8, init_off=.04, init_std=.02)
        graph = Graph(**graph_cfg)
        graph_adj = graph.A.max(0) != 0
        if pooling == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(in_dim))
            # add cls (collects infos from all the other joints, so first and last column are set to all ones)
            graph_adj_cls = torch.ones(graph_adj.shape[0] + 1, graph_adj.shape[1] + 1).bool()
            graph_adj_cls[1:, 1:] = torch.BoolTensor(graph_adj)
            graph_adj = graph_adj_cls
        else:
            # Convert graph_adj to a PyTorch tensor
            graph_adj = torch.BoolTensor(graph_adj)

        A = graph_adj.float()

        self.gcns = GCN(in_dim, embed_dims, A, num_stage=num_stage, p_dropout=drop_rate, adaptive=adaptive, is_bn=True)
        if self.pooling == "weighted_pool":
            self.pool_layer = WeightedPool(embed_dims)

        self.temp_fusion = MultiModalFusion(embed_dims, embed_dims)
        self.final_proj2 = nn.Sequential(
            nn.Conv1d(22 * embed_dims, (embed_dims // 4) * 22, kernel_size=3, padding=1, groups=22, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d((embed_dims // 4) * 22, embed_dims, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25)
        )



    def forward(self, x, temporal_mask, **kwargs):
        bs, t = x.shape[:2]
        if self.pooling == "cls_token":
            cls_token = repeat(self.cls_token, 'd -> b t 1 d', b=bs, t=t)
            # adding the embedding token for all sequences
            x = torch.cat((cls_token, x), dim=2)
        x = rearrange(x, 'b t p d -> (b t) p d')
        
        # Perform interactions between joint points
        x = self.gcns(x)
        
        if self.pooling == "weighted_pool":
            x = self.pool_layer(x, rearrange(temporal_mask, 'b t -> (b t) 1'))
            x = rearrange(x, '(b t) d -> b t d', b=bs)
            return x
        
        #         
        x = rearrange(x, '(b t) p d -> b t p d', b=bs)
        ##################################################
        #B,T,N,C
        #temporal_mask: B,t
        x = self.temp_fusion(x, temporal_mask)
        x = rearrange(x, 'b t p d -> b t (p d)')

        
        x = self.final_proj2(x.permute(0, 2, 1))

        return x.permute(0, 2, 1)



        #####################################
        # if self.pooling == "cls_token":
        #     x = x[:, :, 0, ...]
        # else:
        #     x = reduce(x, 'b t p d -> b t d', self.pooling)
        # # Uncommenting will cause nan tensor to appear in the subsequent LayerNorm layer.
        # # x = mask_logits(x, temporal_mask.unsqueeze(-1))
        # return x

        
