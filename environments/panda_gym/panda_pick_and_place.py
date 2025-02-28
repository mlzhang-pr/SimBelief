from typing import Any, Dict, Tuple

import numpy as np

from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.utils import distance

from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

import gym
import gym.spaces
import gym.utils.seeding




class PickAndPlace(Task):
    def __init__(
        self,
        sim,
        n_tasks=2,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range: float = 0.2,
        goal_z_range: float = 0.1,
        obj_xy_range: float = 0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        # self.np_random, _ = gym.utils.seeding.np_random()

        self.tasks = self.sample_tasks(n_tasks)
        self.goal = self.tasks[0]['target']
        self.object_position = self.tasks[0]['object']
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", self.object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def reset_task(self, idx):
        self.goal = self.tasks[idx]['target']
        self.object_position = self.tasks[idx]['object']
        self.reset()

    def sample_tasks(self, n_tasks):
        tasks = []
        for i in range(n_tasks):
            target = self._sample_goal()
            object_pos = self._sample_object()
            while distance(target, object_pos) < 0.05:
                target = self._sample_goal()
                object_pos = self._sample_object()
            task = {'target': target, 'object': object_pos}
            tasks.append(task)
        return tasks

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        if np.random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal
    
    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = np.random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)


class PandaPickAndPlaceEnv(RobotTaskEnv):

    metadata = {"render_modes": ["human", "rgb_array"]}
    reward_range = (-float('inf'), float('inf'))

    def __init__(
        self,
        n_tasks=2,
        max_episode_steps=50,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse", # 'sparse'
        control_type: str = "ee",
        renderer: str = "Tiny",
    ) -> None:
        
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, n_tasks=n_tasks, reward_type=reward_type)
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        self.task_dim = 3
        self._max_episode_steps = max_episode_steps
        self.observation = self.reset()

        self.observation_space = gym.spaces.Box(-10, 10, shape=self.observation.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=self.robot.action_space.shape, dtype=np.float32)
        self.compute_reward = self.task.compute_reward

        self.success = False
        # self.seed()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velococity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        # achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        desired_goal = self.task.get_goal().astype(np.float32)
        return np.concatenate([observation, desired_goal])
        # return observation
    
    def reset(self):
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        return observation
    
    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self.task.goal
    
    def reset_task(self, idx):
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset_task(idx)
        observation = self._get_obs()
        return observation

    def save_state(self) -> int:
        state_id = self.sim.save_state()
        self._saved_goal[state_id] = self.task.goal
        return state_id

    def restore_state(self, state_id: int) -> None:
        self.sim.restore_state(state_id)
        self.task.goal = self._saved_goal[state_id]


    def remove_state(self, state_id: int) -> None:
        self._saved_goal.pop(state_id)
        self.sim.remove_state(state_id)


    def step(self, action):
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal()))
        done = False
        info = {"is_success": terminated}
        self.success = terminated
        reward = float(self.task.compute_reward(self.task.get_achieved_goal(), self.task.get_goal(), info))
        return observation, reward, done, info
    
    def close(self) -> None:
        self.sim.close()

    def is_goal_state(self):
        return self.success

    def get_all_task_idx(self):
        return range(len(self.task.tasks))
    
    def seed(self, seed=None):
        self.task.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]