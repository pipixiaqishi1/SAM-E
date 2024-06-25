from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
from samE.mvt.segment_anything import sam_model_registry
from samE.mvt.sam_lora_image_encoder import LoRA_Sam, LoRA_Sam_encoder,LoRA_Sam_a


from samE.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    Fusion_up
)

def get_model_para(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0
        
    
    print(f'{model.__class__.__name__}\'s parameter size: {param_size/1024/1024}MB')
    print(f'{model.__class__.__name__}\'s trainable parameter size: {trainable_param_size/1024/1024}MB')
    
    print(f'{model.__class__.__name__}\'s parameter num: {param_num/1000/1000}M')
    print(f'{model.__class__.__name__}\'s trainable parameter num: {trainable_para_num/1000/1000}M')

class MVTSAMlora(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        renderer_device="cuda:0",
        renderer=None,
        action_horizon=None,
        ifSAM=True,
        lora_finetune = True,
        ifsep = True,
        resize_rgb = False
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param add_pixel_loc:
        :param add_depth:
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.act_horizon = action_horizon
        self.ifSAM=ifSAM
        self.lora_finetune = lora_finetune
        self.ifsep = ifsep
        self.resize_rgb = resize_rgb
        if self.ifSAM:
            self.feat_img_dim = 48
            self.sam_img_dim = 48
        else:
            self.feat_img_dim = 96
            self.sam_img_dim = 0
        
        
        print(f"MVT Vars: {vars(self)}")

        if self.ifSAM:
            sam, img_embedding_size = sam_model_registry["vit_b"](image_size=self.img_size if not self.resize_rgb else 320, num_classes=1, checkpoint=None)
            if self.lora_finetune:
                lora_sam = LoRA_Sam(sam, 4)
                self.sam_image_encoder = lora_sam.sam.image_encoder
                get_model_para(self.sam_image_encoder)
            else:
                for param in sam.image_encoder.parameters():
                    param.requires_grad = False
                self.sam_image_encoder = sam.image_encoder
                get_model_para(self.sam_image_encoder)
                

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 320 / 16 = 20

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)   # FIXME  not sure ahout why do this
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1


        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                32,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            7 if self.ifsep else 10,   # 
            self.feat_img_dim,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )
            
        if self.sam_img_dim != 0:
            self.fusion_up = Fusion_up(
                256,
                self.sam_img_dim,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )
            
            self.fusion = Conv2DBlock(
                256,
                self.sam_img_dim,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )
            

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,  # 128
            attn_dim,   # 512
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.im_channels,
            kernel_sizes=self.img_patch_size-1-self.img_patch_size%2,
            strides=self.img_patch_size,
            norm=None,
            activation=activation,
        )

        final_inp_dim = self.im_channels + 10  # 10 : input dim

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.act_horizon*self.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=activation,
        )

        self.trans_decoder = Conv2DBlock(
            self.final_dim,
            1,  
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        feat_out_size = feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.final_dim

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_img * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, feat_out_size),
        )

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """
        # import time;t0 = time.time()
        bs, num_img, img_feat_dim, h, w = img.shape # torch.Size([bs, num_img:5, 7+3, 220, 220])
        
        num_pat_img = h // self.img_patch_size   ## 220//11 = 20 np = 20
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim   10
        assert h == w == self.img_size   # 220

        img = img.view(bs * num_img, img_feat_dim, h, w)  # 10
        d0 = img
        
        if self.ifSAM:  # use SAM
            rgb_img = img[:,3:6]
            
            # rgb_img = img[:,3:6]
            # rgb_clip = True
            # if rgb_clip:
            #     rgb_img[rgb_img<0] = 0
            #     rgb_img[rgb_img>1] = 1 
                
            if self.resize_rgb:
                # import pdb;pdb.set_trace()
                rgb_img = F.interpolate(rgb_img, size=(320, 320), mode='bilinear', align_corners=True)
            sam_out = self.sam_image_encoder(rgb_img)  ## torch.Size([bs*num_img, 256, 220->13, 13])  sam encoder  image patchify, channel: 256
            if num_pat_img == sam_out.shape[-1]:
                rgb_img = self.fusion(sam_out)
            else:
                rgb_img = self.fusion_up(sam_out)    ## c 256-> sam_img_dim
            rgb_img = (
                rgb_img.view(
                bs,
                num_img,
                self.sam_img_dim,
                num_pat_img,
                num_pat_img,
            ).transpose(1, 2).clone())   # torch.Size([bs, 48, 5, 20, 20])
        # concat 
                
        if self.ifsep:
            indices = [0, 1, 2, 6, 7, 8, 9]
            feat_img = img[:,indices]   
        else:
            feat_img = img
                

            
            
        feat_img = self.patchify(feat_img)   # conv2d  c 7 or 10 -> self.feat_img_dim
        feat_img = (
            feat_img.view(
                bs,
                num_img,
                self.feat_img_dim,
                num_pat_img,
                num_pat_img,
            ).transpose(1, 2).clone())   # torch.Size([bs, feat_img_dim :96 or 48, 5, 20, 20])
        _, _, _d, _h, _w = feat_img.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,32]    
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([rgb_img, feat_img, p], dim=1) if self.ifSAM  \
                else torch.cat([feat_img, p], dim=1) # [B, 128, num_img, np, np]   96+32 or 48+48+32 = 128

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding   # learnable  (1, num_img * np * np, 128)

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, 77 + num_img * np * np, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:  # True
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # self attention  among lx and multi image
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128] (3, 5 ,20, 20, 128)
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np] (3, 128 ,5, 20, 20)

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]  # [B, 128, num_img] torch.Size([3, 128, 5])  max-pooling:{f_k}
        _feat = rearrange(_feat, 'b c n -> b (c n)') # torch.Size([3, 640])
        feat.append(repeat(_feat, f'b d -> b {self.act_horizon} d'))  # torch.Size([3, ah, 640])

        x = (  # [B, 128, num_img, np, np]
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
            #  [ B*n, c, np, np] 
            # torch.Size([15, 128, 20, 20]  
        )
        u0 = self.up0(x)                                                            #Conv2DUpsample c:128->64 torch.Size([15, 64, 220, 220])
        u0 = torch.cat([u0, d0], dim=1)                        #  u0:torch.Size([15, 64, 220, 220]) + d0:torch.Size([15, 10, 220, 220]): origin image before attn
        u = self.final(u0)                                          #cond2d  c:128-> act_horizon * 64      torch.Size([3*5, ah*64, 220, 220])   image feature
        
        u = u.view(bs*num_img, self.act_horizon, -1 ,h ,w)   #[3*5, ah, 64 , 220, 220 ]
        u = rearrange(u, 'bn ah c h w -> (bn ah) c h w') # torch.Size([3*5*ah, 64, 220, 220])
        
        # translation decoder
        trans = self.trans_decoder(u).view(bs, num_img, self.act_horizon ,h, w)   #conv2d c:64->1  torch.Size([3, 5, ah, 220, 220])     trans only update by trans

        hm = F.softmax(trans.detach().view(bs, num_img, self.act_horizon ,h * w), -1).view(bs * num_img*self.act_horizon, 1, h * w)      # detach: hm will not be traced by feat， feat update u  TODO: important
        hm = hm.view(bs * num_img*self.act_horizon, 1, h, w)      # torch.Size([3*5*ah, 1, 220, 220])     heatmap feature: softmax over height and width

        _feat = torch.sum(hm * u, dim=[2, 3])  # sum:{f_k*h_k}  torch.Size([3*5*8, 1, 220, 220]) * torch.Size([3*5*8, 64, 220, 220]) ->\
                                                                                    #  torch.Size([3*5*8, 64, 220, 220]) -> (3*5*8, 64)
        _feat = _feat.view(bs, num_img, self.act_horizon, -1)  # torch.Size([3, 5, ah, 64])
        _feat = rearrange(_feat, 'b n ah c -> b ah (n c)')    # torch.Size([3, 5, ah, 64]) ->  torch.Size([3, ah, 5*64]) 
        
        feat.append(_feat)   # [ torch.Size([3, ah, 640])] append torch.Size([3, ah, 320])
        feat = torch.cat(feat, dim=-1)  # torch.Size([3, ah , 960])
        feat = self.feat_fc(feat) # MLP: 960->220   torch.Size([3, ah, 220])
        
        
        out = {"trans": trans, "feat": feat}  # torch.Size([3, 5, ah, 220, 220])  torch.Size([3, ah, 220])  
 
        return out

    def get_wpt(self, q_trans, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = q_trans.shape[0]

        q_trans = q_trans.transpose(1, 2)  # [3, nc, 220*220]
        hm = torch.nn.functional.softmax(q_trans, 2)     # heatmap
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()

    def switch_encoder_trainable(self, trainable=True):
        for param in self.sam_image_encoder.parameters():
            param.requires_grad = trainable
        print(f'Switch self.sam_image_encoder parameters trainable:{trainable}')
        
    def switch_whole_trainable(self, trainable=True):
        for param in self.parameters():
            param.requires_grad = trainable
        print(f'Switch whole parameters trainable:{trainable}')
        
    def add_lora2encoder(self):
        lora_sam_encoder = LoRA_Sam_encoder(self.sam_image_encoder,4)
        self.sam_image_encoder = lora_sam_encoder.sam_encoder
        print(f'add lora to sam encoder and freeze encoder base model')
        