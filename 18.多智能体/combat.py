import gym
from typing import Union
from PIL import Image, ImageDraw
import copy
import logging
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


logger = logging.getLogger(__name__)


class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True




def get_cell_sizes(cell_size: Union[int, list, tuple]):
    """Handle multiple type options of `cell_size`.

    In order to keep the old API of following functions, as well as add
    support for non-square grids we need to check cell_size type and
    extend it appropriately.

    Args:
        cell_size: integer of tuple/list size of two with cell size 
            in horizontal and vertical direction.

    Returns:
        Horizontal and vertical cell size.
    """
    if isinstance(cell_size, int):
        cell_size_vertical = cell_size
        cell_size_horizontal = cell_size
    elif isinstance(cell_size, (tuple, list)) and len(cell_size) == 2:
        # Flipping coordinates, because first coordinates coresponds with height (=vertical direction)
        cell_size_vertical, cell_size_horizontal = cell_size
    else:
        raise TypeError("`cell_size` must be integer, tuple or list with length two.")

    return cell_size_horizontal, cell_size_vertical


def draw_grid(rows, cols, cell_size=50, fill='black', line_color='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)

    width = cols * cell_size_x
    height = rows * cell_size_y
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height

    for x in range(0, image.width, cell_size_x):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=line_color)

    x = image.width - 1
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, cell_size_y):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    y = image.height - 1
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y, x_dash, y_dash = row + margin_x, col + margin_y, row + cell_size_x - margin_x, col + cell_size_y - margin_y
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def write_cell_text(image, text, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y = row + margin_x, col + margin_y
    ImageDraw.Draw(image).text((x, y), text=text, fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size_x, col + cell_size_y)], outline=fill, width=3)


def draw_circle(image, pos, cell_size=50, fill='black', radius=0.3):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    gap_x, gap_y = cell_size_x * radius, cell_size_y * radius
    x, y = row + gap_x, col + gap_y
    x_dash, y_dash = row + cell_size_x - gap_x, col + cell_size_y - gap_y
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_border(image, border_width=1, fill='black'):
    width, height = image.size
    new_im = Image.new("RGB", size=(width + 2 * border_width, height + 2 * border_width), color=fill)
    new_im.paste(image, (border_width, border_width))
    return new_im


def draw_score_board(image, score, board_height=30):
    im_width, im_height = image.size
    new_im = Image.new("RGB", size=(im_width, im_height + board_height), color='#e1e4e8')
    new_im.paste(image, (0, board_height))

    _text = ', '.join([str(round(x, 2)) for x in score])
    ImageDraw.Draw(new_im).text((10, board_height // 3), text=_text, fill='black')
    return new_im


class Combat(gym.Env):
    """
    We simulate a simple battle involving two opposing teams in a n x n grid.
    Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
    square around the team center, which is picked uniformly in the grid. At each time step, an agent can
    perform one of the following actions: move one cell in one of four directions; attack another agent
    by specifying its ID j (there are m attack actions, each corresponding to one enemy agent); or do
    nothing. If agent A attacks agent B, then B’s health point will be reduced by 1, but only if B is inside
    the firing range of A (its surrounding 3 × 3 area). Agents need one time step of cooling down after
    an attack, during which they cannot attack. All agents start with 3 health points, and die when their
    health reaches 0. A team will win if all agents in the other team die. The simulation ends when one
    team wins, or neither of teams win within 100 time steps (a draw).

    The model controls one team during training, and the other team consist of bots that follow a hardcoded policy.
    The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
    it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
    is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

    When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
    encoding its unique ID, team ID, location, health points and cooldown. A model controlling an agent
    also sees other agents in its visual range (3 × 3 surrounding area). The rewards are given agent-wise:
    +1 if the agent hits an opponent, -1 if the agent is hit by an opponent.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(15, 15), n_agents=5, n_opponents=5, init_health=3, full_observable=False,
                 step_cost=0, max_steps=100):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_opponents = n_opponents
        self._max_steps = max_steps
        self._step_cost = step_cost
        self._step_count = None

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5 + self._n_opponents) for _ in range(self.n_agents)])

        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}
        self.opp_pos = {_: None for _ in range(self.n_agents)}
        self.opp_prev_pos = {_: None for _ in range(self.n_agents)}

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        self.opp_health = {_: None for _ in range(self._n_opponents)}
        self._agent_dones = [None for _ in range(self.n_agents)]
        self._agent_cool = {_: None for _ in range(self.n_agents)}
        self._opp_cool = {_: None for _ in range(self._n_opponents)}
        self._total_episode_reward = None
        self.viewer = None
        self.full_observable = full_observable

        # 5 * 5 * (type, id, health, cool, x, y)
        self._obs_low = np.repeat([-1., 0., 0., -1., 0., 0.], 5 * 5)
        self._obs_high = np.repeat([1., n_opponents, init_health, 1., 1., 1.], 5 * 5)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(5)]
            meaning += ['Attack Opponent {}'.format(o) for o in range(self._n_opponents)]
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)
            assert agent_i <= self.n_agents

            return action_meaning[agent_i]
        else:
            return action_meaning

    @staticmethod
    def _one_hot_encoding(i, n):
        x = np.zeros(n)
        x[i] = 1
        return x.tolist()

    def get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its unique ID, team ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
            # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # _agent_i_obs += [self.agent_health[agent_i]]
            # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down

            # team id , unique id, location,health, cooldown

            _agent_i_obs = np.zeros((6, 5, 5))
            for row in range(0, 5):
                for col in range(0, 5):

                    if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                            PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                        x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                        _type = 1 if PRE_IDS['agent'] in x else -1
                        _id = int(x[1:]) - 1  # id
                        _agent_i_obs[0][row][col] = _type
                        _agent_i_obs[1][row][col] = _id
#                         print('type', type, '_type', _type)
                        _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]
                        _agent_i_obs[3][row][col] = self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]
                        _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1  # cool/uncool

                        _agent_i_obs[4][row][col] = pos[0] / self._grid_shape[0]  # x-coordinate
                        _agent_i_obs[5][row][col] = pos[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)

        return _obs

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __init_full_obs(self):
        """ Each team consists of m = 10 agents and their initial positions are sampled uniformly in a 5 × 5
        square.
        """
        self._full_obs = self.__create_grid()

        # select agent team center
        # Note : Leaving space from edges so as to have a 5x5 grid around it
        agent_team_corner = random.randint(0, int(self._grid_shape[0] / 2)), random.randint(0, int(self._grid_shape[1] / 2))
        agent_pos_index = random.sample(range(25), self.n_agents)
        # randomly select agent pos
        for agent_i in range(self.n_agents):
            pos = [int(agent_pos_index[agent_i] / 5) + agent_team_corner[0], agent_pos_index[agent_i] % 5 + agent_team_corner[1]]
#             print(pos)
            while True:
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break
                pos = [random.randint(agent_team_corner[0], agent_team_corner[0] + 4),
                       random.randint(agent_team_corner[1], agent_team_corner[1] + 4)]

        # select opponent team center
        while True:
            pos = random.randint(agent_team_corner[0], self._grid_shape[0] - 5), random.randint(agent_team_corner[1], self._grid_shape[1] - 5)
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                opp_team_corner = pos
                break
                
        opp_pos_index = random.sample(range(25), self._n_opponents)

        # randomly select opponent pos
        for opp_i in range(self._n_opponents):
            pos = [int(opp_pos_index[agent_i] / 5) + opp_team_corner[0], opp_pos_index[agent_i] % 5 + opp_team_corner[1]]
            while True:
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.opp_prev_pos[opp_i] = pos
                    self.opp_pos[opp_i] = pos
                    self.__update_opp_view(opp_i)
                    break
                pos = [random.randint(opp_team_corner[0], opp_team_corner[0] + 4),
                       random.randint(opp_team_corner[1], opp_team_corner[1] + 4)]

        self.__draw_base_img()

    def reset(self):
        self._step_count = 0
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.__init_full_obs()
        return self.get_agent_obs()

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        # draw agents
        for agent_i in range(self.n_agents):
            if self.agent_health[agent_i] > 0:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        # draw opponents
        for opp_i in range(self._n_opponents):
            if self.opp_health[opp_i] > 0:
                fill_cell(img, self.opp_pos[opp_i], cell_size=CELL_SIZE, fill=OPPONENT_COLOR)
                write_cell_text(img, text=str(opp_i + 1), pos=self.opp_pos[opp_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __update_opp_pos(self, opp_i, move):

        curr_pos = copy.copy(self.opp_pos[opp_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.opp_pos[opp_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_opp_view(opp_i)

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    @staticmethod
    def is_visible(source_pos, target_pos):
        """
        Checks if the target_pos is in the visible range(5x5)  of the source pos

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    @staticmethod
    def is_fireable(source_pos, target_pos):
        """
        Checks if the target_pos is in the firing range(3x3)

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) \
               and (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)

    def reduce_distance_move(self, source_pos, target_pos):

        # Todo: makes moves Enum
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append('UP')
        elif source_pos[0] < target_pos[0]:
            _moves.append('DOWN')

        if source_pos[1] > target_pos[1]:
            _moves.append('LEFT')
        elif source_pos[1] < target_pos[1]:
            _moves.append('RIGHT')

        move = random.choice(_moves)
        for k, v in ACTION_MEANING.items():
            if move.lower() == v.lower():
                move = k
                break
        return move

    @property
    def opps_action(self):
        """
        Opponent bots follow a hardcoded policy.

        The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
        it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
        is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

        :return:
        """

        visible_agents = set([])
        opp_agent_distance = {_: [] for _ in range(self._n_opponents)}

        for opp_i, opp_pos in self.opp_pos.items():
            for agent_i, agent_pos in self.agent_pos.items():
                if agent_i not in visible_agents and self.agent_health[agent_i] > 0 \
                        and self.is_visible(opp_pos, agent_pos):
                    visible_agents.add(agent_i)
                distance = abs(agent_pos[0] - opp_pos[0]) + abs(agent_pos[1] - opp_pos[1])  # manhattan distance
                opp_agent_distance[opp_i].append([distance, agent_i])

        opp_action_n = []
        for opp_i in range(self._n_opponents):
            action = None
            for _, agent_i in sorted(opp_agent_distance[opp_i]):
                if agent_i in visible_agents:
                    if self.is_fireable(self.opp_pos[opp_i], self.agent_pos[agent_i]):
                        action = agent_i + 5
                    else:
                        action = self.reduce_distance_move(self.opp_pos[opp_i], self.agent_pos[agent_i])
                    break
            if action is None:
                logger.info('No visible agent for enemy:{}'.format(opp_i))
                action = random.choice(range(5))
            opp_action_n.append(action)


        return opp_action_n

    def step(self, agents_action):
        assert len(agents_action) == self.n_agents

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # What's the confusion?
        # What if agents attack each other at the same time? Should both of them be effected?
        # Ans: I guess, yes
        # What if other agent moves before the attack is performed in the same time-step.
        # Ans: May be, I can process all the attack actions before move directions to ensure attacks have their effect.

        # processing attacks
        agent_health, opp_health = copy.copy(self.agent_health), copy.copy(self.opp_health)
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action > 4:  # attack actions
                    target_opp = action - 5
                    if self.is_fireable(self.agent_pos[agent_i], self.opp_pos[target_opp]) \
                            and opp_health[target_opp] > 0:
                        opp_health[target_opp] -= 1
                        rewards[agent_i] += 1

                        if opp_health[target_opp] == 0:
                            pos = self.opp_pos[target_opp]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']

        opp_action = self.opps_action
        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                target_agent = action - 5
                if action > 4:  # attack actions
                    if self.is_fireable(self.opp_pos[opp_i], self.agent_pos[target_agent]) \
                            and agent_health[target_agent] > 0:
                        agent_health[target_agent] -= 1
                        rewards[target_agent] -= 1

                        if agent_health[target_agent] == 0:
                            pos = self.agent_pos[target_agent]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']

        self.agent_health, self.opp_health = agent_health, opp_health

        # process move actions
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action <= 4:
                    self.__update_agent_pos(agent_i, action)

        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                if action <= 4:
                    self.__update_opp_pos(opp_i, action)
        win = False
        # step overflow or all opponents dead
        if (self._step_count >= self._max_steps) \
                or (sum([v for k, v in self.opp_health.items()]) == 0) \
                or (sum([v for k, v in self.agent_health.items()]) == 0):
            self._agent_dones = [True for _ in range(self.n_agents)]
            if (sum([v for k, v in self.opp_health.items()]) == 0):
                win = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'health': self.agent_health, 'win': win}

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
OPPONENT_COLOR = 'blue'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': 'E',
    'agent': 'A',
    'opponent': 'X',
}
