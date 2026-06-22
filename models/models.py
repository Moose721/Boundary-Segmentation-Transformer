import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import drop_path, to_2tuple, trunc_normal_


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class PatchEmbed(nn.Module):
    def __init__(self, num_frames, img_size=224, tubelet_size=8, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        num_spatial_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        num_temporal_patches = num_frames // tubelet_size

        self.num_spatial_patches = num_spatial_patches
        self.num_temporal_patches = num_temporal_patches
        self.num_patches = num_spatial_patches * num_temporal_patches

        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels=in_chans, 
            out_channels=embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        assert x.shape[1] == self.num_patches, f"Mismatch in num_patches"
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

'''
Window attention module
'''
class Attention(nn.Module):
    def __init__(self, 
                dim, 
                num_heads, 
                attn_drop=0.,
                proj_drop = 0.,
                mask=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(in_features=dim, out_features=3*all_head_dim, bias=False)

        self.proj = nn.Linear(in_features=all_head_dim, out_features=dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask

    def forward(self, x):
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=self.qkv.bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = q @ k.transpose(-2, -1) if self.mask is None else q @ k.transpose(-2, -1) + self.mask
 
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BSWAttention(nn.Module):
    def __init__(self,
                dim,
                num_heads,
                qkv_bias=False,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.,
                block_size=196,
                window_factor=0,
                ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(in_features=dim, out_features=3*all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.block_size = block_size
        self.window_factor = window_factor

    def forward(self, x):
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=self.qkv.bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        num_groups = N // (self.block_size+1) #block + class_token
        window_ratio = self.window_factor*2 + 1 #number of blocks to attend to centered on current 
        pad_const = (self.block_size+1)*self.window_factor
        unfold_size = (self.block_size+1)*window_ratio
    
        q = q.reshape(B, self.num_heads, num_groups, self.block_size+1, -1)
       
        k = F.pad(k, (0, 0, pad_const, pad_const))
        k = k.unfold(dimension=2, size=unfold_size, step=self.block_size+1)
        v = F.pad(v, (0, 0, pad_const, pad_const))
        v = v.unfold(dimension=2, size=unfold_size, step=self.block_size+1).transpose(-2, -1)
    
        q = q * self.scale
        window_attn = q @ k
        window_attn = window_attn.softmax(dim=-1)
        window_attn = self.attn_drop(window_attn)
        
        x = (window_attn @ v).reshape(B, self.num_heads, N, -1)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                dim, 
                num_heads, 
                mlp_ratio=4., 
                qkv_bias=False,
                qk_scale=None,
                attn_drop=0.,
                drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                block_size=196,
                window_factor=0, 
                ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        self.attn = BSWAttention(
            dim=dim, 
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            block_size=block_size,
            window_factor=window_factor
        )

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            out_features=dim
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



def inject_class_tokens(N, group_size, embed_dim=768):
    remainder = N % group_size
    
    pad_len = group_size - remainder
    x = F.pad(x, (0, 0, 0, pad_len))


    num_groups = x.shape[1] // group_size
    x = x.reshape(-1, num_groups, group_size, embed_dim)
    cls_tokens = nn.Parameter(torch.randn(B, num_groups, 1, embed_dim))
    pos_embed = nn.Parameter(torch.zeros(1, num_groups, 1, embed_dim))
    temp_embed = nn.Parameter(torch.zeros(1, num_groups, 1, embed_dim))
    cls_tokens = cls_tokens + pos_embed + temp_embed

    x = torch.cat((cls_tokens, x), dim=2)
    x = x.reshape(B, -1, embed_dim)
    return x


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

#temp = get_sinusoid_encoding_table(100, 768)
#print(temp.shape)

def get_attention_mask(self, seq_len, window_size, window_factor, stride):
    """
    Generates a sliding window attention mask.
    
    Args:
        seq_len (int): The length of the sequence.
        window_size (int): The attention span (number of previous tokens allowed).
        shift (int): Number of tokens to shift the window. Default is 0.
                    Positive shift moves the window into future tokens.
    
    Returns:
        torch.Tensor: A boolean or float attention mask where True (or 0) 
                    allows attention and False (or -inf) masks it out.
    """
    # Create a 2D grid where rows and columns represent indices
    indices = torch.arange(seq_len)
    row_idx = indices.unsqueeze(1)
    col_idx = indices.unsqueeze(0)
    

    # Calculate valid bounds for the sliding window
    mask = torch.zeros(seq_len*seq_len).reshape(seq_len, seq_len)

    lower_bound = row_idx 
    upper_bound = row_idx + window_size
    for i in row_idx:
        r_slice = torch.zeros(seq_len)
        window_idx = i // window_size #determines which window current row index falls into
        start_idx = max(0, (window_idx - window_factor) * window_size)
        end_idx = min(seq_len, (window_idx + window_factor + 1) * window_size )
        
        
    # Create the mask
    #mask = (col_idx >= lower_bound) & (col_idx < upper_bound)
    #print(mask)
    return mask


def temp_function(embed_dim):
    return nn.Parameter(torch.zeros(167, embed_dim))

class Encoder(nn.Module):
    def __init__(self, 
                 num_frames,
                 img_size=224, 
                 tubelet_size=8,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_heads=8,
                 mlp_ratio=4.,    
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate = 0.,
                 drop_path_rate=0.,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 use_learnable_pos_embed=False,
                 block_size=196,
                 window_factor=0,
                 depth=12,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.block_size = block_size
        self.patch_embed = PatchEmbed(
            num_frames=num_frames,
            img_size=img_size,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_spatial_patches = self.patch_embed.num_spatial_patches
        num_temporal_patches = self.patch_embed.num_temporal_patches
        seq_len = self.patch_embed.num_patches

        self.num_spatial_patches = num_spatial_patches

        if use_learnable_pos_embed:
            self.pos_spatial_embed = nn.Parameter(torch.zeros(1, num_spatial_patches, embed_dim))
            self.pos_temporal_embed = nn.Parameter(torch.zeros(1, num_temporal_patches, embed_dim))
        else:
            self.pos_spatial_embed = get_sinusoid_encoding_table(num_spatial_patches, embed_dim//2)
            self.pos_temporal_embed = get_sinusoid_encoding_table(num_temporal_patches, embed_dim//2)
        
        #block_cls_tokens, pad_len, num_blocks = get_block_tokens(seq_len, block_size, embed_dim)

        remainder = seq_len % block_size
        pad_len = block_size - remainder if remainder > 0 else 0
        num_blocks = (seq_len + pad_len) // block_size

        self.pad_len = pad_len
        self.num_blocks = num_blocks

        block_cls_tokens = nn.Parameter(torch.zeros((num_blocks, embed_dim)))
        block_pos_embed = nn.Parameter(torch.zeros((num_blocks, embed_dim)))
        block_temp_embed = nn.Parameter(torch.zeros((num_blocks, embed_dim)))

        self.blk_cls_tokens = block_cls_tokens
        self.blk_pos_embed = block_pos_embed
        self.blk_temp_embed = block_temp_embed

        self.pos_spatial_drop = nn.Dropout(drop_rate)
        self.pos_temporal_drop = nn.Dropout(drop_rate)
       
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        wf = [depth//2 + 1 - abs(i - depth//2) for i in range(depth)]
        
        self.blocks = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                block_size=block_size,
                window_factor=wf[i],
            ) 
            for i in range(depth)
        ])
        

    def forward(self, x):
        #patch embed
        x = self.patch_embed(x)
        
        #add spatial and temporal positional encodings
        B, N, C = x.shape
        num_spatial_groups = N // self.num_spatial_patches
        x = x.reshape(B, num_spatial_groups, self.num_spatial_patches, -1)

        pos_spatial_embed = F.pad(self.pos_spatial_embed, (0, self.embed_dim//2))
        pos_temporal_embed = F.pad(self.pos_temporal_embed, (self.embed_dim//2, 0))
        
        x = x + pos_spatial_embed.unsqueeze(1).type_as(x).to(x.device).clone().detach()
        x = self.pos_spatial_drop(x)

        x = x + pos_temporal_embed.unsqueeze(2).type_as(x).to(x.device).clone().detach()
        x = self.pos_temporal_drop(x)
        
        x = x.reshape(B, -1, C)

        x = F.pad(x, (0, 0, 0, self.pad_len))
        x = x.reshape(B, self.num_blocks, self.block_size, -1)

        blk_tokens = self.blk_cls_tokens + self.blk_pos_embed + self.blk_temp_embed
        blk_tokens = blk_tokens.view(1, self.num_blocks, 1, C)
        blk_tokens = blk_tokens.expand(B, -1, -1, -1).type_as(x).to(x.device)
        
        x = torch.cat((blk_tokens, x), dim=2)
        x = x.reshape(B, -1, C)
      
        for blck in self.blocks:
            x = blck(x) 
        
        return x
        

class Decoder(nn.Module):
    def __init__(self,
                embed_dim=768,
                block_size=196,
                num_classes=2
                ):
        super().__init__()
        self.block_size = block_size
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward(self, x):
        B, N, C = x.shape
        
        cls_tokens = x[:, ::self.block_size+1, :]
        
        output = self.head(cls_tokens)
        return output

class BoundaryTransformer(nn.Module):
    def __init__(self,
                 num_frames,
                 img_size=224, 
                 tubelet_size=8,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_heads=8,
                 mlp_ratio=4.,    
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 attn_drop=0.,
                 norm_layer=nn.LayerNorm,
                 use_learnable_pos_embed=False,
                 block_size=196,
                 window_factor=0,
                 num_classes=2,
                 depth=12,
                 ):
        super().__init__()

        self.encoder = Encoder(
            num_frames=num_frames,
            img_size=img_size, 
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,    
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate = drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            use_learnable_pos_embed=use_learnable_pos_embed,
            block_size=block_size,
            window_factor=window_factor,
            depth=depth,
        )

        self.decoder = Decoder(
            embed_dim=embed_dim,
            block_size=block_size,
            num_classes=num_classes
        )
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def boundary_transformer_base():
    model = BoundaryTransformer(
        num_frames=256,
        block_size=392,
        num_classes=2,
        depth=13,
    )
    
    return model


#input = torch.rand((3, 3, 256, 224, 224))
#model = boundary_transformer_base()
#output = model(input)
#print(output.shape)
#for name, param in model.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Trainable: {param.requires_grad}")


