from uuid import uuid4
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import math
import operator
import functools
import numpy as np

class EntropyController(object):
    entropy = 1.0
    adjustables = []

    @classmethod
    def register(cls, adjustable):
        EntropyController.adjustables.append(adjustable)

    @classmethod
    def set_entropy(cls, value):
        for adjustable in EntropyController.adjustables:
            adjustable.set_entropy(value)

class AdjustableEntropyMixin(object):

    def __init__(self):
        EntropyController.register(self)
        self.entropy = 1.0

    def set_entropy(self, value):
        self.entropy = value

class MomentumWalkingMixin(object):
    def __init__(self):
        self.heading = 0
        self.momentum = .2
        self.movement_noise = .2
        self.attractions = []
        self.attraction_amounts = {}

    def _angle_to_agent(self, agent):
        x = self.pos[0]
        y = self.pos[1]
        x_other = agent.pos[0]
        y_other = agent.pos[1]
        return math.atan2(y_other - y, x_other - x)

    def new_heading(self):
        prev_heading = self.heading
        noise = random.gauss(0, self.movement_noise)

        self.clean_attractions()
        if self.attractions:
            total_force = 0
            for attraction_amount in self.attraction_amounts.values():
                total_force += attraction_amount
            choice_probabilities = [self.attraction_amounts[attraction.unique_id]/total_force for attraction in self.attractions]
            focused_attraction = np.random.choice(self.attractions, 1, p=choice_probabilities)[0]
            self.heading = self._angle_to_agent(focused_attraction) + noise
        else:
            self.heading = random.gauss(prev_heading, math.pi * self.momentum) + noise

    def clean_attractions(self):
        to_remove = []
        for attraction in self.attractions:
            if not attraction.active or self.attraction_amounts[attraction.unique_id] <= 0:
                to_remove.append(attraction)
        for a in to_remove:
            self.remove_attraction(a)

    def remove_attraction(self, agent):
        del self.attraction_amounts[agent.unique_id]
        self.attractions.remove(agent)

    def get_step(self):
        grid = self.model.grid
        cart_x = round(math.cos(self.heading))
        cart_y = round(math.sin(self.heading))
        actual_x, actual_y = grid.torus_adj((cart_x + self.pos[0], cart_y + self.pos[1]))
        return (actual_x, actual_y)

    def attract(self, agent, attraction_amount=.2):
        self.attractions.append(agent)
        self.attraction_amounts[agent.unique_id] = attraction_amount

    def walk(self):
        self.new_heading()
        next_pos = self.get_step()
        self.model.grid.move_agent(self, next_pos)

class BaseAgent(Agent):

    def __init__(self, model):
        unique_id = "{0}_{1}".format(self.__class__.__name__, \
            uuid4())
        super().__init__(unique_id, model)
        self.active = True

    def status_str(self):
        return self.unique_id

class KillableAgent(BaseAgent):

    def __init__(self, model):
        super().__init__(model)
        self.alive = True

    def kill(self):
        self.alive = False
        self.active = False
        self.model.remove(self)

    def status_str(self):
        return "{0} | alive: {1}".format(
            super().status_str(),
            self.alive
        )

class Consumable(BaseAgent):
    def __init__(self, model):
        super().__init__(model)
        self.nutrition = 10
        self.consumed = False

    def consume(self):
        self.consumed = True
        self.active = False
        self.model.remove(self)


class SludgeFood(Consumable):
    def __init__(self, model):
        super().__init__(model)
        self._init_attribs()

    def _init_attribs(self):
        self.nutrition = random.randrange(100)


class SludgeMonster(KillableAgent, AdjustableEntropyMixin, MomentumWalkingMixin):
    def __init__(self, model, parents=None):
        AdjustableEntropyMixin.__init__(self)
        MomentumWalkingMixin.__init__(self)
        super().__init__(model)
        self.health = 100
        self.parents = parents
        self.leader = None
        self.food_sought = None

        if parents:
            assert type(parents) is tuple and len(parents) == 2

        self._init_attribs()

    def _init_attribs(self):
        if not self.parents:
            self.friendliness = random.uniform(.19, .21)
            self.anger = random.uniform(.19, .21)
            self.fertility = random.uniform(.22, .23)
            self.max_attack = random.randint(0,50)
            self.max_hug_benefit = random.randint(0,20)
            self._decay_mult = random.random() * 30 + 40
            self.leadership = random.random()
            self.follower_mult = random.random()
            self.leader_attraction = random.uniform(.19, .21)
            self.food_attraction = random.uniform(.30, .50)
            self.sight = random.randrange(10)
            self.movement_noise = random.uniform(.20,.50)
        else:
            self.friendliness = self._mix_attribs("friendliness", .1, min_val=0, max_val=1)
            self.anger = self._mix_attribs("anger", .1, min_val=0, max_val=1)
            self.fertility = self._mix_attribs("fertility", .1, min_val=0, max_val=1)
            self.max_attack = self._mix_attribs("max_attack", 10, min_val=0, max_val=100, isint=True)
            self.max_hug_benefit = self._mix_attribs("max_hug_benefit", 3, min_val=0, max_val=30, isint=True)
            self._decay_mult = self._mix_attribs("_decay_mult", 10, min_val=30, max_val=100)
            self.leadership = self._mix_attribs("leadership", .1, min_val=0, max_val=1)
            self.follower_mult = self._mix_attribs("follower_mult", .1, min_val=0, max_val=1)
            self.leader_attraction = self._mix_attribs("leader_attraction", .1, min_val=0, max_val=0)
            self.food_attraction = self._mix_attribs("food_attraction", .1, min_val=0, max_val=0)
            self.sight = self._mix_attribs("sight", 1, min_val=0, max_val=10, isint=True)
            self.movement_noise = self._mix_attribs("movement_noise", .1, min_val=0, max_val=1)
            print("friendliness: {0} anger: {1} fertility: {2} max_attack: {3} max_hug_benefit: {4}"
                " _decay_mult: {5}".format(self.friendliness, self.anger, self.fertility, self.max_attack,
                    self.max_hug_benefit, self._decay_mult))

    def _mix_attribs(self, _attrib_name, std, min_val=None, max_val=None, isint=False):
        parent_a_val = self.parents[0].__dict__[_attrib_name]
        parent_b_val = self.parents[1].__dict__[_attrib_name]
        mean = (parent_a_val + parent_b_val) / 2
        if random.random() > .95:
            mean += random.gauss(0, std)

        attrib = random.gauss(mean, std)
        if min_val is not None:
            attrib = max(attrib, min_val)
        if max_val is not None:
            attrib = min(attrib, max_val)
        if isint:
            attrib = int(attrib)
        return attrib

    def _decay(self):
        overpopulation_effect = 100/(1+math.exp(-1 * ((self.pop_modifier()*20) - 10)))
        #print(overpopulation_effect)
        return self.entropy * self._decay_mult + overpopulation_effect

    def step(self):
        #print(self._decay() * self.pop_modifier())
        self.health -= self._decay() * self.pop_modifier()
        if self.health <= 0:
            self.kill()
        else:
            self.move()
            self.interact()

    def move(self):
        # possible_steps = self.model.grid.get_neighborhood(
        #     self.pos,
        #     moore=True,
        #     include_center=False
        # )
        # print(possible_steps)
        # next_position = random.choice(possible_steps)
        self.walk()

    def interact(self):
        self.clean_leader()
        self.clean_sought_food()
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates.remove(self)
        sludgemates = []
        foodmates = []
        for cellmate in cellmates:
            if isinstance(cellmate, SludgeMonster):
                sludgemates.append(cellmate)
            elif isinstance(cellmate, SludgeFood):
                foodmates.append(cellmate)

        for sludgemate in sludgemates:
            follow_prob = self.follower_mult * sludgemate.leadership
            if follow_prob >  random.random():
                self.add_leader(sludgemate)

        perform_hug = random.random() > .5
        if len(sludgemates) > 0:
            other = random.choice(sludgemates)
            sludgemates.remove(other)
            if perform_hug:
                self.hug(other)
            else:
                perform_hug = True
                self.attack(other)
        if len(sludgemates) > 0:
            if perform_hug:
                self.hug(other)
            else:
                self.attack(other)

        self.feed(foodmates)
        self.seek_food()

    def seek_food(self):
        if not self.food_sought:
            neighbors = self.model.grid.get_neighbors(
                 self.pos,
                 moore=True,
                 include_center=False,
                 radius=self.sight
            )

            closet_food_distance = None
            closet_food = None
            for neighbor in neighbors:
                if isinstance(neighbor, Consumable):
                    this_dist = ((self.pos[0] - neighbor.pos[0])**2 + (self.pos[1] - neighbor.pos[1])**2)**(1/2)
                    if closet_food_distance is None or closet_food_distance > this_dist:
                        closet_food = neighbor
                        closet_food_distance = this_dist
            if(closet_food):
                self.add_sought_food(closet_food)

    def clean_sought_food(self):
        if self.food_sought and self.food_sought.consumed:
            self.food_sought = None

    def add_sought_food(self, food):
        if self.food_sought and self.food_sought in self.attractions:
            self.remove_attraction(self.food_sought)
        self.food_sought = food
        self.attract(self.food_sought, self.food_attraction)

    def clean_leader(self):
        if self.leader and not self.leader.alive:
            self.leader = None

    def add_leader(self, leader):
        if self.leader and self.leader in self.attractions:
            self.remove_attraction(self.leader)
        self.leader = leader
        self.attract(leader, self.leader_attraction)

    @property
    def is_following(self):
        return self.leader is not None


    def pop_modifier(self):
        space = self.model.width * self.model.height
        return self.model.num_agents/space

    def feed(self, nearby_food):
        if nearby_food:
            self.consume(random.choice(nearby_food))

    def consume(self, food):
        self.health += food.nutrition
        food.consume()

    def attack(self, other):
        if self.leader == other:
            attraction_prob_mod = 0.1
        else:
            attraction_prob_mod = 1.0
        prob = self.anger * self.pop_modifier() * attraction_prob_mod
        if prob > random.random() and self.max_attack > 0:
            attack_val = random.randrange(self.max_attack)
            other.health -= attack_val
            self.health = min(self.health + (attack_val * 0.5), 100)

    def hug(self, other):
        prob = self.friendliness * (-1/self.pop_modifier())
        if prob > random.random() and self.max_hug_benefit > 0:
            other.health += random.randrange(self.max_hug_benefit)
        if self.fertility * (self.health/100.0) > random.random():
            self.reproduce(other)

    def reproduce(self, other):
        offspring = SludgeMonster(self.model, parents=(self, other))
        own_neighborhood = self.model.grid.get_neighborhood(
             self.pos,
             moore=True,
             include_center=False
        )
        other_neighborhood = self.model.grid.get_neighborhood(
            other.pos,
            moore=True,
            include_center=False
        )
        intersection = [cell for cell in own_neighborhood if cell in other_neighborhood]
        placement = random.choice(intersection)
        agent = self.model.add_agent(agent=offspring, pos=placement)

    def status_str(self):
        return "{0} | health: {1} | entropy {2}".format(
            super().status_str(),
            self.health,
            self.entropy
        )


class SludgeMonsterModel(Model):


    def __init__(self, num_agents, width=100, height=100, food_growth_prob=0.0005, initial_food_growth=.30, collection_frequency=1):
        self.running = True
        self.width = width
        self.height = height
        self.food_growth_prob = food_growth_prob
        self.initial_food_growth = initial_food_growth
        self.food_type = SludgeFood
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, True)
        self.datacollector = DataCollector(model_reporters={
            "friendliness": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="friendliness"),
            "anger": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="anger"),
            "fertility": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="fertility"),
            "max_attack": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="max_attack"),
            "max_hug_benefit": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="max_hug_benefit"),
            "_decay_mult": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="_decay_mult"),
            "leadership": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="leadership"),
            "follower_mult": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="follower_mult"),
            "leader_attraction": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="leader_attraction"),
            "food_attraction": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="food_attraction"),
            "sight": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="sight"),
            "movement_noise": lambda m: m.average_agent_val(agent_type=SludgeMonster, attrib_name="movement_noise"),
            "is_following": lambda m: m.average_agent_val(agent_type=SludgeMonster, func=lambda a:a.is_following)
            })

        self.collection_frequency = collection_frequency
        self.num_agents = num_agents
        for i in range(self.num_agents):
            self.add_agent()

        self.grow_food(self.initial_food_growth)

    def average_agent_val(self, agent_type, attrib_name=None, func=None):
        vals = []
        agent_count = 0
        if attrib_name:
            for agent in self.schedule.agents:
                if isinstance(agent, agent_type):
                    vals.append(getattr(agent, attrib_name))
                    agent_count += 1
        elif func:
            for agent in self.schedule.agents:
                if isinstance(agent, agent_type):
                    vals.append(func(agent))
                    agent_count += 1
        else:
            raise Exception("bad params")
        if agent_count > 0:
            return functools.reduce(operator.add, vals)/float(agent_count)
        else:
            return 0

    def step(self):
        if self.schedule.steps % self.collection_frequency == 0:
            self.datacollector.collect(self)
        self.schedule.step()
        self.grow_food(self.food_growth_prob)

    def grow_food(self, growth_prob):
        area = self.height * self.width
        food_growth_areas = np.random.random((self.height, self.width)) < (growth_prob)
        existing_food = self.get_agent_locations(self.food_type)
        food_growth_areas = np.logical_and(food_growth_areas, np.logical_not(existing_food))
        for y in range(self.height):
            for x in range(self.width):
                if food_growth_areas[y,x]:
                    self.add_agent(agent=self.food_type(self), pos=(x,y))

    def get_agent_locations(self, agent_type):
        truth_table = np.zeros((self.height, self.width), np.bool)
        for contents, x, y in self.grid.coord_iter():
            if any([isinstance(agent, agent_type) for agent in contents]):
                truth_table[y, x] = True
        return truth_table

    def remove(self, agent):
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)

    def add_agent(self, agent=None, pos=None):
        if not pos:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            pos = (x,y)
        if not agent:
            agent = SludgeMonster(self)
        self.schedule.add(agent)
        self.grid.place_agent(agent, pos)
        self.num_agents = len(self.schedule.agents)
        return agent

    def status_str(self):
        val = [self.__class__.__name__]
        for agent in self.schedule.agents:
            val.append(agent.status_str())
        if len(self.schedule.agents) == 0:
            val.append("No agents in model.")
        return "\n".join(val)


# model = SludgeMonsterModel(100)
# EntropyController.set_entropy(.1)

# plt.ion()

# for i in range(100):
#     model.step()
#     status = model.status_str()
#     longest_line = max([len(line) for line in status.split("\n")])
#     space = max(0, longest_line - 2)
#     sep = "#" + "-" * space + "#"
#     print(sep)
#     print(status)
#     print(sep)
#     print()

#     visualization = np.zeros((model.grid.width, model.grid.height))
#     for agent in model.schedule.agents:
