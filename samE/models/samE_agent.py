import pprint

import torch
import torchvision
import numpy as np
import torch.nn as nn

import clip
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import samE.utils.peract_utils as peract_utils
import samE.mvt.utils as mvt_utils
import samE.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils
from samE.mvt.augmentation import apply_se3_aug_con_sequence, aug_utils, apply_se3_aug_given_matrix
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from samE.utils.dataset import _clip_encode_text
from samE.utils.lr_sched_utils import GradualWarmupScheduler


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}
    
    
    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val),0)
    pprint.pprint(out)
    return out

def manage_hm_log(
    agent,
    hm_log,
    reset_log,
):
    if not hasattr(agent, "hm_log") or reset_log:
        agent.hm_log = {}
    
    for key, val in hm_log.items():
        agent.hm_log[key] = val

def save_hm_log(agent,path):
    for key, val in agent.hm_log.items():
        for i, img in enumerate(val):
            img.save(f'{path}{key}_{i}.jpg')
            
            
def horizon_mask(output, data_horizon):

    shape = output.shape
    bs, ah = shape[0], shape[1]
    mask = torch.arange(ah).expand(bs, ah).to(output.device) < data_horizon.view(bs, 1)
    if len(shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(output)

    output = output * mask

    return output
    
def horizon_loss_cal(output, data_horizon):

    shape = output.shape
    bs, ah = shape[0], shape[1]
    mask = torch.arange(ah).expand(bs, ah).to(output.device) < data_horizon.view(bs, 1)
    if len(shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(output)

        output_mask = output * mask
        ah_ave = output_mask.sum(dim=(0,2))/(mask.sum(dim=(0,2)).float() + 1e-5)
        
    else:
        output_mask = output * mask
        ah_ave = output_mask.sum(dim=0)/(mask.sum(dim=0).float() + 1e-5)
        
    total_ave = output_mask.sum()/(mask.sum().float())
    
    return total_ave,ah_ave
 
            
class samEAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        add_lang: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        log_dir="",
        action_horizon=None,
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound

        self.action_horizon = action_horizon
        
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if self._optimizer_type == "lamb":
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b]
            action_collision_one_hot[b, gt_ignore_collisions] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        if len(dims) == 4:
            # no action chunking
            bs, nc, h, w = dims
            assert isinstance(only_pred, bool)

            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2) # [3, 220*220, nc]
            if not only_pred:
                q_trans = q_trans.clone()

            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
            y_q = None

            return q_trans, rot_q, grip_q, collision_q, y_q, pts
        
        elif len(dims) == 5:
            # action chunking
            bs, nc, ah, h, w = dims
            assert isinstance(only_pred, bool)

            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc,  ah, h * w).transpose(1, 3)   # torch.Size([3,220*220, ah, nc])
            if not only_pred:
                q_trans = q_trans.clone()

            # (bs, 218)
            rot_q = out["feat"].view(bs,  ah, -1)[:, 
                                                 :, 
                                                 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs,  ah, -1)[:, 
                                                  :, 
                                                  self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs,  ah, -1)[:, 
                                                       :, 
                                                       self.num_all_rot + 2 : self.num_all_rot + 4]
            y_q = None

            return q_trans, rot_q, grip_q, collision_q, y_q, pts
            

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        
        ah = self.action_horizon
        # sample
        tasks = replay_sample["tasks"]
        
        action_rot_grip = replay_sample["rot_grip_action_indicies_list"].int()  # (b, max_ah, 4) of int
        action_grip = action_rot_grip[:,:, -1]  # (b,max_ah)
        
        action_gripper_pose = replay_sample["gripper_pose_list"]  # (b, max_ah, 7)
        # action_trans_con = action_gripper_pose[:, :ah, 0:3]  # (b, ah, 3)
        # action_rot = action_gripper_pose[:, :ah, 3:7]  # (b, ah, 4)   rotation in quaternion xyzw
        
        action_ignore_collisions = replay_sample["ignore_collisions_list"].int()  # (b, max_ah, 1) of int
        obs_ignore_collisions = replay_sample["obs_ignore_collisions"].int()  # (b, 1) of int
        obs_ignore_collisions = obs_ignore_collisions.reshape(-1) # (b)
        
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()

        # import pdb;pdb.set_trace()

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)    #  4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
        return_out = {}

        data_horizon = replay_sample["horizon"]
        
        with torch.no_grad():
            wpt_local_chunk=[]
            action_rot_list=[]
             
            obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)   # [rgb, pcd], [pcd]
            pc, img_feat = rvt_utils.get_pc_img_feat(obs,pcd)  #  pc = point_cloud (bs,128*128*camera_nums,3)   img_feat=rgb  (bs,128*128*camera_nums,3)
            
            pcd = pc.detach().clone()
            
            for i in range(ah):  
                if self._transform_augmentation and backprop:   # True
                    action_gripper_pose_s = action_gripper_pose[:,i]  # 
                    
                    aug = True
                    if aug:
                        if i == 0:
                            action_trans_con, action_rot, pc, trans_shift_4x4, rot_shift_4x4 = apply_se3_aug_con_sequence(  # TODO  augmentation: wpt,rot from action_gripper_pose_s,    pc augmentation also
                                pcd=pcd,                                             # action_trans_con:(bs,3)  action_rot:(bs,4)  pc:(bs,128*128*4,3)
                                action_gripper_pose=action_gripper_pose_s,
                                bounds=torch.tensor(self.scene_bounds),
                                trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                                rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                                pc_aug=True
                            )
                             
                        else:
                            # import pdb;pdb.set_trace()
                            action_trans_con, action_rot= apply_se3_aug_given_matrix(
                                pcd=pcd,                                             
                                action_gripper_pose=action_gripper_pose_s,
                                bounds=torch.tensor(self.scene_bounds),
                                trans_shift_4x4 = trans_shift_4x4,
                                rot_shift_4x4 = rot_shift_4x4,
                            ) 
                            
                        action_trans_con = torch.tensor(action_trans_con).to(pcd.device)
                        action_rot = torch.tensor(action_rot).to(pcd.device)
                        
                    else:
                        action_trans_con = action_gripper_pose_s[:, :3]
                        action_rot = action_gripper_pose_s[:, 3:7]
                    
                        action_trans_con = torch.tensor(action_trans_con).to(pcd.device)
                        action_rot = torch.tensor(action_rot).to(pcd.device)

                    if i == 0:
                        pc, img_feat = rvt_utils.move_pc_in_bound(  # remove point out of bound from pc and corresponding point from img_feat
                                                    pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound) 
                
                # TODO: vectorize
                action_rot = action_rot.cpu().numpy()   
                for j, _action_rot in enumerate(action_rot):
                    _action_rot = aug_utils.normalize_quaternion(_action_rot)
                    if _action_rot[-1] < 0:
                        _action_rot = -_action_rot
                    action_rot[j] = _action_rot
                action_rot_list.append(action_rot)

                wpt = [x[:3] for x in action_trans_con]  # wpt = action_trans_con
                wpt_local = []
                rev_trans = []
                for _pc, _wpt in zip(pc, wpt):  #  calculate the transformation according to pc_s0
                    a, b = mvt_utils.place_pc_in_cube(
                        _pc,
                        _wpt,
                        with_mean_or_bounds=self._place_with_mean,   # False     place to scene_bounds
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    wpt_local.append(a.unsqueeze(0))
                    rev_trans.append(b)

                wpt_local = torch.cat(wpt_local, axis=0)
                wpt_local_chunk.append(wpt_local)



            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None
        
        out = self._network(        #  pc = point_cloud (bs,128*128*camera_nums,3)   img_feat=rgb  (bs,128*128*camera_nums,3)   camera_nums = 4
            pc=pc,
            img_feat=img_feat,   
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=img_aug,
        )

        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, ah, h, w)    # q_trans:torch.Size([3,220*220, ah, nc]) rot_q:(bs, ah,72*3) grip_q:(3, ah,2) collision_q:(3, ah,2)
                                             
        )
        
        action_rot_x_one_hot = []
        action_rot_y_one_hot = []
        action_rot_z_one_hot = []
        action_grip_one_hot = []
        action_collision_one_hot = []
        action_trans_chunk = []
        for i in range(ah):
            action_rot_i = action_rot_list[i]
            action_grip_i = action_grip[:,i]
            
            action_ignore_collisions_i = obs_ignore_collisions if i == 0 else action_ignore_collisions[:,i-1]
            (
                action_rot_x_one_hot_i,       # (bs, 72)
                action_rot_y_one_hot_i,       # (bs, 72)
                action_rot_z_one_hot_i,       # (bs, 72)
                action_grip_one_hot_i,        # (bs, 2)
                action_collision_one_hot_i,   # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot_i, action_grip_i, action_ignore_collisions_i, device=self._device)   # label
            action_rot_x_one_hot.append(action_rot_x_one_hot_i.unsqueeze(1))   # append (bs, 1, 72)
            action_rot_y_one_hot.append(action_rot_y_one_hot_i.unsqueeze(1))
            action_rot_z_one_hot.append(action_rot_z_one_hot_i.unsqueeze(1))
            action_grip_one_hot.append(action_grip_one_hot_i.unsqueeze(1))
            action_collision_one_hot.append(action_collision_one_hot_i.unsqueeze(1))
        
            
            action_trans = self.get_action_trans(
                wpt_local_chunk[i], pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            )  # label    wpt:(bs,3)  ->  action_trans heat_map: (bs,220*220,5)     heatmap has been normalized: sum=1 
            action_trans_chunk.append(action_trans.unsqueeze(2)) # append (3,220*220,1,nc)
           
         
        action_rot_x_one_hot = torch.cat(action_rot_x_one_hot,1)
        action_rot_y_one_hot = torch.cat(action_rot_y_one_hot,1)
        action_rot_z_one_hot = torch.cat(action_rot_z_one_hot,1)
        action_grip_one_hot = torch.cat(action_grip_one_hot,1)
        action_collision_one_hot = torch.cat(action_collision_one_hot,1)    
        action_trans_chunk = torch.cat(action_trans_chunk,2)   
        
        
        ah_mask = True
        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss_list = self._cross_entropy_loss(q_trans, action_trans_chunk) # 
            if ah_mask:
                # import pdb;pdb.set_trace()
                trans_loss, trans_loss_ah = horizon_loss_cal(trans_loss_list, data_horizon)
            else:
                trans_loss = trans_loss_list.mean()
                trans_loss_ah = torch.mean(trans_loss_list,(0,2))
            
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                rot_loss_x_list = self._cross_entropy_loss(
                    rot_q[:,:,0 * self._num_rotation_classes : 1 * self._num_rotation_classes,].transpose(1, 2),
                    action_rot_x_one_hot.argmax(-1),)
                if ah_mask:
                    rot_loss_x,rot_loss_x_ah = horizon_loss_cal(rot_loss_x_list, data_horizon)
                else:
                    rot_loss_x = rot_loss_x_list.mean()
                    rot_loss_x_ah = torch.mean(rot_loss_x_list,0)
                
                
                rot_loss_y_list = self._cross_entropy_loss(
                    rot_q[:,:,1 * self._num_rotation_classes : 2 * self._num_rotation_classes,].transpose(1, 2),
                    action_rot_y_one_hot.argmax(-1),)
                if ah_mask:
                    rot_loss_y, rot_loss_y_ah = horizon_loss_cal(rot_loss_y_list, data_horizon)
                else:
                    rot_loss_y = rot_loss_y_list.mean()
                    rot_loss_y_ah = torch.mean(rot_loss_y_list,0)
                
                rot_loss_z_list = self._cross_entropy_loss(
                    rot_q[:,:,2 * self._num_rotation_classes : 3 * self._num_rotation_classes,].transpose(1, 2),action_rot_z_one_hot.argmax(-1),)
                if ah_mask:
                    rot_loss_z, rot_loss_z_ah = horizon_loss_cal(rot_loss_z_list, data_horizon)
                else:
                    rot_loss_z = rot_loss_z_list.mean()
                    rot_loss_z_ah = torch.mean(rot_loss_z_list,0)
                
                grip_loss_list = self._cross_entropy_loss(grip_q.transpose(1, 2), action_grip_one_hot.argmax(-1),)
                if ah_mask:
                    grip_loss, grip_loss_ah = horizon_loss_cal(grip_loss_list, data_horizon)
                else:
                    grip_loss = grip_loss_list.mean()
                    grip_loss_ah = torch.mean(grip_loss_list,0)

                collision_loss_list = self._cross_entropy_loss(
                    collision_q.transpose(1, 2), action_collision_one_hot.argmax(-1))
                if ah_mask:
                    collision_loss, collision_loss_ah = horizon_loss_cal(collision_loss_list, data_horizon)
                else:
                    collision_loss = collision_loss_list.mean()
                    collision_loss_ah = torch.mean(collision_loss_list,0)

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
                       
            self._optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self._optimizer.step()
            self._lr_sched.step()

            total_loss_ah = (
                trans_loss_ah
                + rot_loss_x_ah
                + rot_loss_y_ah
                + rot_loss_z_ah
                + grip_loss_ah
                + collision_loss_ah
            )
            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            
            ah_log={}
            for i in range(ah):
                ah_log[f'total_loss_ah_{i}'] = total_loss_ah.tolist()[i]
                ah_log[f'trans_loss_ah_{i}'] = trans_loss_ah.tolist()[i]
            
            # print(ah_log)
            loss_log.update(ah_log)
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out['ah_log']=ah_log
            
            
            
        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        ah = self.action_horizon

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )
        q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, ah, h, w), only_pred=True   # TODO  action_horizon
        )
        action_chunking=[]
        for i in range(ah):
            q_trans_i = q_trans[:,:,i]
            rot_q_i = rot_q[:,i]
            grip_q_i = grip_q[:,i]
            collision_q_i = collision_q[:,i]
            pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            q_trans_i, rot_q_i, grip_q_i, collision_q_i, y_q, rev_trans, dyn_cam_info)
            continuous_action = np.concatenate(
                (
                    pred_wpt[0].cpu().numpy(),
                    pred_rot_quat[0],
                    pred_grip[0].cpu().numpy(),
                    pred_coll[0].cpu().numpy(),
                )
            )
            action_chunking.append(ActResult(continuous_action))
        return action_chunking

    def get_pred(
        self,
        q_trans,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        pred_wpt_local = self._net_mod.get_wpt(q_trans, dyn_cam_info, y_q)

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(              
            wpt_local.unsqueeze(1), dyn_cam_info=dyn_cam_info, out=None
        )   # pt_img of size (bs, np=1, num_img, 2)
        assert wpt_img.shape[1] == 1
        
        wpt_img = wpt_img.squeeze(1)  # (bs, num_img, 2)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
        
        