from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
import random
from gym.utils import seeding
import numpy as np

class FourRoomsEnvRaw(MiniGridEnv):


    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100, n_tasks=2,seed=None,**kwargs):
        # self._agent_default_pos = agent_pos
        # self._goal_default_pos = goal_pos

        self.size = 17
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation   先指定agent的位置
        # if self._agent_default_pos is not None:
        #     self.agent_pos = self._agent_default_pos
        #     self.grid.set(*self._agent_default_pos, None)
        #     # assuming random start direction
        #     self.agent_dir = self._rand_int(0, 4)
        # else:
        #     self.place_agent()

        # if self._goal_default_pos is not None:
        #     goal = Goal()
        #     self.put_obj(goal, *self._goal_default_pos)
        #     goal.init_pos, goal.cur_pos = self._goal_default_pos
        # else:
        #     self.place_obj(Goal())


class FourRoomsEnv(MiniGridEnv):  # for muti-task  generate task id
    def __init__(self, agent_pos=(2,2), goal_pos=None, max_steps=100, n_tasks=2,seed=None,**kwargs):
        self.num_tasks = n_tasks
        # self._agent_default_pos = agent_pos
        # self._goal_default_pos = self._goal

        self.size = 17
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )


        self.states = [(x, y) for y in np.arange(0, self.size) for x in np.arange(0, self.size)]
        self.possible_goals = self.states.copy()
       
        self.num_tasks = min(n_tasks, len(self.possible_goals))

        self.goals = random.sample(self.possible_goals, self.num_tasks)   # 抽取num——tasks个

        # 判断goals的选择是否合理
        for goal_pos in self.goals:
             # Don't place the object on top of another object
            if self.grid.get(*goal_pos) is not None:
                self.goals.remove(goal_pos)

            # Don't place the object where the agent is
            if np.array_equal(goal_pos, self.agent_pos):
                self.goals.remove(goal_pos)


        if seed is not None:
            self.seed(seed)

        self._max_episode_steps = max_steps

        self.reset_task(0) # index==0


    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

         # Randomize the player start position and orientation   先指定agent的位置
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]
    
    def get_all_task_idx(self):
        return range(len(self.goals))
    
    def get_task(self):
        return self._goal

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def reset_task(self, idx=None):  # 设置goal pos
        ' reset goal and state '
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self._agent_default_pos =(2,2)
        self._goal_default_pos = self._goal
        
        
        # if self._agent_default_pos is not None:
        #     self.agent_pos = self._agent_default_pos
        #     self.grid.set(*self._agent_default_pos, None)
        #     # assuming random start direction
        #     self.agent_dir = self._rand_int(0, 4)
        # else:
        #     self.place_agent()

        # if self._goal_default_pos is not None:
        #     goal = Goal()
        #     self.put_obj(goal, *self._goal_default_pos)
        #     goal.init_pos, goal.cur_pos = self._goal_default_pos
        # else:
        #     self.place_obj(Goal())

        self.reset()







    























