from uuid import uuid4
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random
import matplotlib.pyplot as plt

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

class BaseAgent(Agent):

    def __init__(self, model):
        unique_id = "{0}_{1}".format(self.__class__.__name__, \
            uuid4())
        super().__init__(unique_id, model)

    def status_str(self):
        return self.unique_id

class KillableAgent(BaseAgent):

    def __init__(self, model):
        super().__init__(model)
        self.alive = True

    def kill(self):
        self.alive = False
        self.model.remove(self)

    def status_str(self):
        return "{0} | alive: {1}".format(
            super().status_str(),
            self.alive
        )

class SludgeMonster(KillableAgent, AdjustableEntropyMixin):
    def __init__(self, model):
        AdjustableEntropyMixin.__init__(self)
        super().__init__(model)
        self.health = 100

        self.friendliness = 1.0
        self.anger = 0.3
        self.fertility = .5

    def _decay(self):
        return self.entropy * 10

    def step(self):
        print(self._decay() * self.pop_modifier())
        self.health -= self._decay() * self.pop_modifier()
        if self.health <= 0:
            self.kill()
        else:
            self.move()
            self.interact()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        next_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, next_position)

    def interact(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates.remove(self)

        perform_hug = random.random() > .5
        if len(cellmates) > 0:
            other = random.choice(cellmates)
            cellmates.remove(other)
            if perform_hug:
                self.hug(other)
            else:
                perform_hug = True
                self.attack(other)
        if len(cellmates) > 0:
            if perform_hug:
                self.hug(other)
            else:
                self.attack(other)

    def pop_modifier(self):
        space = self.model.width * self.model.height
        return self.model.num_agents/space

    def attack(self, other):
        prob = self.anger * self.pop_modifier()
        chance = prob > random.random()
        if chance:
            other.health -= random.randrange(10)

    def hug(self, other):
        prob = self.friendliness * (-1/self.pop_modifier())
        chance = prob > random.random()
        if chance:
            other.health += random.randrange(10)
        chance = self.fertility > random.random()
        if chance:
            self.model.add_agent()

    def status_str(self):
        return "{0} | health: {1} | entropy {2}".format(
            super().status_str(),
            self.health,
            self.entropy
        )


class SludgeMonsterModel(Model):
    def __init__(self, num_agents, width=100, height=100):
        self.running = True
        self.width = width
        self.height = height
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, True)

        self.num_agents = num_agents
        for i in range(self.num_agents):
            self.add_agent()

    def step(self):
        self.schedule.step()

    def remove(self, agent):
        self.schedule.remove(agent)
        self.grid.remove_agent(agent)

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
