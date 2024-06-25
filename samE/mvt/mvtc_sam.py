import copy
import torch

from torch import nn

import samE.mvt.utils as mvt_utils


from samE.mvt.mvt_sam_lora import MVTSAMlora
from samE.mvt.config import get_cfg_defaults
from samE.mvt.renderer import BoxRenderer


class MVTC_Sam(nn.Module):
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
        action_horizon=None,
        renderer_device="cuda:0",
        ifSAM=True,
        lora_finetune=True,
        ifsep=True,
        resize_rgb = False
    ):
        """MultiView Transfomer"""
        super().__init__()

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]

        # for verifying the input
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.proprio_dim = proprio_dim
        self.img_size = img_size
        
        self.action_horizon = action_horizon
        
        self.mvt1 = MVTSAMlora(**args, renderer=self.renderer)

        

    def get_pt_loc_on_img(self, pt, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, np, x = pt.shape
        assert x == 3
        assert out is None
        out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        wpt = self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        return wpt

    def render(self, pc, img_feat, img_aug, dyn_cam_info):  #  render from point cloud and image
        mvt = self.mvt1
        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info
            
            # import pdb;pdb.set_trace()
            if mvt.add_corr:   # True
                img = [
                    self.renderer(
                        _pc,         # (num_points, 3)  x,y,z
                        torch.cat((_pc, _img_feat), dim=-1),  # (num_points, 6)   (x,y,z,r,g,b)
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

            img = torch.cat(img, 0)  # （bs,5,220,220,7）  x,y,z,r,g,b,d
            img = img.permute(0, 1, 4, 2, 3)  # [bs, 5, 7, 220, 220]

            # for visualization purposes
            if mvt.add_corr:
                mvt.img = img[:, :, 3:].clone().detach()
            else:
                mvt.img = img.clone().detach()

            # image augmentation     add noise to image
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if mvt.add_pixel_loc:    # true  
                bs = img.shape[0]
                pixel_loc = mvt.pixel_loc.to(img.device)   # to read
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )

        return img   # torch.Size([bs, 5, 7+3, 220, 220])    10 =  6+1+3

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
    ):
        if not self.training:
            # no img_aug when not training
            assert img_aug == 0

        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug)   # pc:   bs, (num_points, 3)   img_feat: bs, (num_points, 3)    all the points from 4 cameras
        # import pdb;pdb.set_trace()
        img = self.render(
            pc,
            img_feat,
            img_aug,
            dyn_cam_info=None,
        )
        
        out = self.mvt1(img=img, proprio=proprio, lang_emb=lang_emb, **kwargs)
        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()
