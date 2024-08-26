import os
import yaml
import csv
import torch
import cv2
import shutil

import numpy as np

from omegaconf import OmegaConf
from multiprocessing import Value
from tensorflow.python.summary.summary_iterator import summary_iterator
from copy import deepcopy

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator_sequence
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.log_writer import LogWriter
from yarr.agents.agent import VideoSummary

import samE.mvt.config as default_mvt_cfg
import samE.models.samE_agent as samE_agent
import samE.config as default_exp_cfg

from samE.mvt.mvtc_sam import MVTC_Sam
from samE.libs.peract.helpers import utils
from samE.utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from samE.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    get_official_peract,
)
from samE.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)
from samE.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
)
from samE.utils.rvt_utils import load_agent as load_agent_state

from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor


def get_model_size(model):
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

def load_agent(
    model_path=None,
    peract_official=False,
    peract_model_dir=None,
    exp_cfg_path=None,
    mvt_cfg_path=None,
    eval_log_dir="",
    device=0,
    use_input_place_with_mean=False,
):
    device = f"cuda:{device}"

    if not (peract_official):
        assert model_path is not None

        # load exp_cfg
        model_folder = os.path.join(os.path.dirname(model_path))

        exp_cfg = default_exp_cfg.get_cfg_defaults()
        if exp_cfg_path != None:
            exp_cfg.merge_from_file(exp_cfg_path)
        else:
            exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))

        # WARNING NOTE: a temporary hack to use place_with_mean in evaluation
        if not use_input_place_with_mean:
            exp_cfg.rvt.place_with_mean = True

        exp_cfg.freeze()
        ACTION_HORIZON = exp_cfg.ah
        
        # create agent
        
        if exp_cfg.agent == "our":
            mvt_cfg = default_mvt_cfg.get_cfg_defaults()
            if mvt_cfg_path != None:
                mvt_cfg.merge_from_file(mvt_cfg_path)
            else:
                mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))

            mvt_cfg.freeze()

            samE = MVTC_Sam(
                action_horizon=ACTION_HORIZON,
                renderer_device=device,
                **mvt_cfg,
            )
            get_model_size(samE)
            agent = samE_agent.samEAgent(
                network=samE.to(device),
                action_horizon = ACTION_HORIZON,
                image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
                add_lang=mvt_cfg.add_lang,
                scene_bounds=SCENE_BOUNDS,
                cameras=CAMERAS,
                log_dir=f"{eval_log_dir}/eval_run",
                **exp_cfg.peract,
                **exp_cfg.rvt,
            )
        else:
            raise NotImplementedError

        agent.build(training=False, device=device)
        load_agent_state(model_path, agent)
        agent.eval()

    elif peract_official:  # load official peract model, using the provided code
        try:
            model_folder = os.path.join(os.path.abspath(peract_model_dir), "..", "..")
            train_cfg_path = os.path.join(model_folder, "config.yaml")
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        except FileNotFoundError:
            print("Config file not found, trying to load again in our format")
            train_cfg_path = "configs/peract_official_config.yaml"
            agent = get_official_peract(train_cfg_path, False, device, bs=1)
        agent.load_weights(peract_model_dir)
        agent.eval()

    print("Agent Information")
    print(agent)
    return agent


@torch.no_grad()
def eval(
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
    record_enabled=False
):
    agent.eval()
    if isinstance(agent, samE_agent.samEAgent):
        agent.load_clip()

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions","total_agent_forward_counts"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    if True:
        rollout_generator = RolloutGenerator_sequence(device)
    else:
        rollout_generator = RolloutGenerator_c(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    if record_enabled:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        cam = VisionSensor.create([1280, 720])
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)

        speed = 0.005
        speed = 0

        
        cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), speed)
        tr = TaskRecorder(eval_env, cam_motion, fps=30)

        eval_env.env._action_mode.arm_action_mode.set_callable_each_step(tr.take_snap)




    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)
    
    execute_horizon = agent.action_horizon
    print(f"execute_horizon: {execute_horizon}")
    scores = []
    for task_id in range(num_tasks):
        if record_enabled:
            tr._cam_motion.save_pose()
        
        task_rewards = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=record_enabled,
                replay_ground_truth=replay_ground_truth,
                execute_horizon=execute_horizon,
            )
            
            try:
                for i,replay_transition in enumerate(generator):
                    if i == 0:
                        print(f" Evaluating {tasks[task_id]} | Episode {ep} | Lang Goal: {eval_env._lang_goal}")
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            agent_forward_count = episode_rollout[-1].agent_forward_count
            if verbose:
                # print(
                #     f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal} | agent_forward_count: {agent_forward_count}"
                # )
                print(
                    f" | Score: {reward} | Episode Length: {len(episode_rollout)} | agent_forward_count: {agent_forward_count}"
                )
            
            if record_enabled:
                hr_video_folder = os.path.join(log_dir, "videos")
                os.makedirs(hr_video_folder, exist_ok=True)
                if reward > 0.99:
                    print(f"###########Success at task {tasks[task_id]}!###########")
                        
                    record_file = os.path.join(hr_video_folder,f'{task_name}_Episode{ep}_success.mp4')
                else:
                    record_file = os.path.join(hr_video_folder,f'{task_name}_Episode{ep}_fail.mp4')
                    
                if reward > 0.99:
                    tr.save(record_file, lang_goal, reward)
                    
                # tr.save(record_file, lang_goal, reward)
                tr._cam_motion.restore_pose()

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions","total_agent_forward_counts"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    elif s.name == "eval_envs/total_agent_forward_counts":
                        csv_results["total_agent_forward_counts"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            video_image_folder = "./tmp"
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            record_folder,
                            f"{task_name}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            record_folder, f"{task_name}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1
                    video_cnt += 1
                    os.makedirs(video_image_folder, exist_ok=True)
                    for idx in range(len(video) - 10):
                        cv2.imwrite(
                            os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                        )
                    images_path = os.path.join(video_image_folder, r"%d.png")
                    os.system(
                        "ffmpeg -i {} -vf palettegen palette.png -hide_banner -loglevel error".format(
                            images_path
                        )
                    )
                    os.system(
                        "ffmpeg -framerate {} -i {} -i palette.png -lavfi paletteuse {} -hide_banner -loglevel error".format(
                            record_fps, images_path, video_path
                        )
                    )
                    os.remove("palette.png")
                    shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    if logging:
        csv_fp.close()

    # set agent to back train mode
    agent.train()

    # unloading clip to save memory
    if isinstance(agent, samE_agent.samEAgent):
        agent.unload_clip()
        agent._network.free_mem()

    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    model_paths = []
    if not (args.peract_official):
        assert args.model_name is not None
        model_paths.append(os.path.join(args.model_folder, args.model_name))
    else:
        model_paths.append(None)

    # skipping evaluated models
    if args.skip:
        """
        to_skip: {
            0: {'light_bulb_in': False, .....}
            1: {'light_bulb_in': False, .....}
            .
            .
        }
        """
        to_skip = {
            get_model_index(x): {y: False for y in args.tasks} for x in model_paths
        }

        filenames = os.listdir(args.eval_log_dir)
        for filename in filenames:
            if not filename.startswith("events.out.tfevents."):
                continue
            summ = summary_iterator(f"{args.eval_log_dir}/{filename}")
            # skipping the time log of the summary
            try:
                next(summ)
            except:
                # moving to the next file
                continue
            for cur_summ in summ:
                cur_task = cur_summ.summary.value[0].tag[5:]
                cur_step = cur_summ.step
                if cur_step in to_skip:
                    to_skip[cur_step][cur_task] = True

    tb = TensorboardManager(args.eval_log_dir)
    for model_path in model_paths:
        tasks_to_eval = deepcopy(args.tasks)

        if args.peract_official:
            model_idx = 0
        else:
            model_idx = get_model_index(model_path)
            if model_idx is None:
                model_idx = 0

        if args.skip:
            for _task in args.tasks:
                if to_skip[model_idx][_task]:
                    tasks_to_eval.remove(_task)

            if len(tasks_to_eval) == 0:
                print(f"Skipping model_idx={model_idx} for args.tasks={args.tasks}")
                continue

        if not (args.peract_official):
            agent = load_agent(
                model_path=model_path,
                exp_cfg_path=args.exp_cfg_path,
                mvt_cfg_path=args.mvt_cfg_path,
                eval_log_dir=args.eval_log_dir,
                device=args.device,
                use_input_place_with_mean=args.use_input_place_with_mean,
            )

            agent_eval_log_dir = os.path.join(
                args.eval_log_dir, os.path.basename(model_path).split(".")[0]
            )
        else:
            agent = load_agent(
                peract_official=args.peract_official,
                peract_model_dir=args.peract_model_dir,
                device=args.device,
                use_input_place_with_mean=args.use_input_place_with_mean,
            )
            agent_eval_log_dir = os.path.join(args.eval_log_dir, "final")

        os.makedirs(agent_eval_log_dir, exist_ok=True)
        scores = eval(
            agent=agent,
            tasks=tasks_to_eval,
            eval_datafolder=args.eval_datafolder,
            start_episode=args.start_episode,
            eval_episodes=args.eval_episodes,
            episode_length=args.episode_length,
            replay_ground_truth=args.ground_truth,
            device=args.device,
            headless=args.headless,
            logging=True,
            log_dir=agent_eval_log_dir,
            verbose=True,
            save_video=args.save_video,
            record_enabled=args.record
        )
        print(f"model {model_path}, scores {scores}")
        task_scores = {}
        for i in range(len(tasks_to_eval)):
            task_scores[tasks_to_eval[i]] = scores[i]

        print("save ", task_scores)
        tb.update("eval", model_idx, task_scores)
        tb.writer.flush()

    tb.close()


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.6, 0)
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    if not (args.peract_official):
        args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    else:
        args.eval_log_dir = os.path.join(args.peract_model_dir, "eval", args.log_name)

    os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)