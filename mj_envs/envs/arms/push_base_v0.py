""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.envs import env_base
from mj_envs.utils.quat_math import euler2quat, mat2euler
from mj_envs.utils.inverse_kinematics import qpos_from_site_pose
from mujoco_py import load_model_from_path, MjSim

class PushBaseV0(env_base.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp', 'qv', 'grasp_pos', 'object_err', 'target_err'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "object_dist": -1.0,
        "target_dist": -1.0,
        "bonus": 4.0,
        "penalty": -50,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
               robot_site_name,
               object_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip=40,
               reward_mode="dense",
               obs_keys=DEFAULT_OBS_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               init_qpos=None,
               pos_limit_low=None,
               pos_limit_high = None,
               object_init_perturb=None,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.object_site_name = object_site_name
        self.object_sid = self.sim.model.site_name2id(object_site_name)
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.jnt_low = self.sim.model.jnt_range[:self.sim.model.nu, 0]
        self.jnt_high = self.sim.model.jnt_range[:self.sim.model.nu, 1]
        assert not ((pos_limit_high is None) ^ (pos_limit_low is None)) # make sure either both are None or neither are None
        self.pos_limit_low = np.array(pos_limit_low)
        self.pos_limit_high = np.array(pos_limit_high)
        self.object_init_perturb = object_init_perturb
        self.ik_sim = MjSim(self.sim.model)
        self.last_ctrl = None
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        if init_qpos is not None:
            self.init_qpos[:len(init_qpos)] = np.array(init_qpos)[:]

        if pos_limit_low is not None:
            act_low = -np.ones(self.pos_limit_low.shape[0]) if self.normalize_act else self.pos_limit_low.copy()
            act_high = np.ones(self.pos_limit_high.shape[0]) if self.normalize_act else self.pos_limit_high.copy()
            self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([self.sim.data.time])
        obs_dict['qp'] = sim.data.qpos.copy()
        obs_dict['qv'] = sim.data.qvel.copy()
        obs_dict['grasp_pos'] = sim.data.site_xpos[self.grasp_sid]
        obs_dict['object_err'] = sim.data.site_xpos[self.object_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['target_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.object_sid]
        return obs_dict


    def get_reward_dict(self, obs_dict):
        object_dist = np.linalg.norm(obs_dict['object_err'], axis=-1)
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)
        far_th = 1.25

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('object_dist',   object_dist),
            ('target_dist',   target_dist),
            ('bonus',   (object_dist<.1) + (target_dist<.1) + (target_dist<.05)),
            ('penalty', (object_dist>far_th)),
            # Must keys
            ('sparse',  1.0*(target_dist<.050)),
            ('solved',  target_dist<.050),
            ('done',    object_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self):
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]
        
        reset_qpos = self.init_qpos.copy()
        obj_jid = self.sim.model.joint_name2id(self.object_site_name)
        reset_qpos[obj_jid:obj_jid+3] += self.np_random.uniform(low=self.object_init_perturb['low'], high=self.object_init_perturb['high'])

        #self.init_qpos[:9] = np.array([0.4653,  0.5063,  0.0228, -2.1195, -0.6052,  0.7064 , 2.5362,  0.025 ,  0.025])
        obs = super().reset(reset_qpos, self.init_qvel)
        return obs

    def step(self, a):

        if a.flatten().shape[0] == self.sim.model.nu:
            act_low = -np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,0].copy()
            act_high = np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,1].copy()
            action = np.clip(a, act_low, act_high)
        else:
            #print('Joints: {}'.format(self.sim.data.qpos[:9]))
            #print('Position: {}'.format(self.sim.data.site_xpos[self.grasp_sid]))

            assert (a.flatten().shape[0] == 7)
            eef_cmd = (0.5 * a.flatten() + 0.5) * (self.pos_limit_high - self.pos_limit_low) + self.pos_limit_low
            if self.pos_limit_low is not None and self.pos_limit_high is not None:
                eef_cmd = np.clip(eef_cmd, self.pos_limit_low, self.pos_limit_high)

            eef_pos = eef_cmd[:3]
            eef_elr = eef_cmd[3:6]

            #eef_pos = np.array([0, 0.35, 1.25])
            #eef_elr = np.array([0.35,3.14,1.57])

            eef_quat = euler2quat(eef_elr)

            self.ik_sim.data.qpos[:self.sim.model.nu] = self.sim.data.qpos[:self.sim.model.nu]
            self.ik_sim.forward()
            ik_result = qpos_from_site_pose(physics=self.ik_sim,
                                            site_name=self.sim.model.site_id2name(self.grasp_sid),
                                            target_pos=eef_pos,
                                            target_quat=eef_quat,
                                            inplace=False,
                                            regularization_strength=1.0,
                                            is_hardware=self.robot.is_hardware)
            action = ik_result.qpos[:self.sim.model.nu]
            action[7:9] = self.init_qpos[7:9]

            if self.normalize_act:
                action = 2 * (((action - self.jnt_low) / (self.jnt_high - self.jnt_low)) - 0.5)

        #print(self.sim.data.site_xpos[self.grasp_sid])
        #print(mat2euler(self.sim.data.site_xmat[self.grasp_sid].reshape(3,3)))
        #exit()
        '''
        action = (0.5 * action + 0.5) * (self.jnt_high - self.jnt_low) + self.jnt_low
        action = np.zeros(9)
        eef_pos = np.array([-0.35,0.48,0.825])
        eef_elr = np.array([3.14,0,0])
        eef_quat = euler2quat(eef_elr)

        self.ik_sim.data.qpos[:7] = np.random.normal(self.sim.data.qpos[:7], 0.0)
        self.ik_sim.data.qpos[2] = 0.0
        self.ik_sim.forward()
        ik_result = qpos_from_site_pose(physics=self.ik_sim,
                                        site_name=self.sim.model.site_id2name(self.grasp_sid),
                                        target_pos=eef_pos,
                                        target_quat=eef_quat,
                                        inplace=False,
                                        regularization_strength=1.0,
                                        is_hardware=self.robot.is_hardware)
        action = ik_result.qpos[:self.sim.model.nu]
        action[7:9] = self.init_qpos[7:9]
        print(action)
        action = 2 * (((action - self.jnt_low) / (self.jnt_high - self.jnt_low)) - 0.5)
        '''

        self.last_ctrl = self.robot.step(ctrl_desired=action,
                                        ctrl_normalized=self.normalize_act,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info
