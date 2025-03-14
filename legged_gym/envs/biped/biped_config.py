# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym.envs.base.base_config import BaseConfig


class BipedCfg(BaseConfig):
    class env:
        num_envs = 8192
        num_observations = 30
        num_height_samples = 117
        num_actions = 6
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5 + 4  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1
        zero_command_prob = 0.0

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            # lin_vel_y = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14159, 3.14159]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.0, 0.1]

    class init_state:
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control:
        action_scale = 0.25

        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42,
            "hip_L_Joint": 42,
            "knee_L_Joint": 42,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 42,
            "hip_R_Joint": 42,
            "knee_R_Joint": 42,
            "foot_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "foot_R_Joint": 0.0,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/PF_TRON1A/urdf/robot.urdf"
        name = "biped"
        foot_name = "foot"
        foot_radius = 0.03
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.0, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 5]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.5
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # termination related rewards
            termination = -0.0
            keep_balance = 1.0
            fail = -0.0

            # tracking related rewards
            tracking_lin_vel = 1
            tracking_ang_vel = 0.5
            tracking_lin_vel_pb = 1.0 * 0
            tracking_ang_vel_pb = 0.2 * 0

            # regulation related rewards
            base_height = -2
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -1
            action_smooth = -0.01
            orientation = -10.0
            feet_distance = -100
            feet_regulation = -0.05
            foot_landing_vel = -0.15
            tracking_contacts_shaped_force = -2
            tracking_contacts_shaped_vel = -2

            # non used rewards
            torque_limits = -0.1 * 0  # lead to unstable training
            power = -2e-5 * 0.0
            feet_slip = -0.05 * 0
            feet_stumble = -0.0
            double_contact = -0.05 * 0
            feet_contact_forces = -0.002 * 0
            no_fly = -2 * 0

            # only work on terrain group
            nominal_state = -0.01 * 0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.68 # 0.58
        feet_height_target = 0.10
        min_feet_distance = 0.115
        about_landing_threshold = 0.08
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class BipedCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = BipedCfg.env.num_observations * BipedCfg.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 15000  # number of policy updates

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_PF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "biped"
        run_name = ""
        # load and resume
        resume = False
        load_run = "-1"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = "None"  # updated from load_run and chkpt
