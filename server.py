from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from agent_model import SludgeMonsterModel, EntropyController, SludgeMonster, SludgeFood
import PIL
import numpy as np

def health_color(value):
    green = int(255 - ((100 - value) * 255/100.0))
    red = 255 - green

    g_code = hex(green).split("x")[-1].rjust(2, "0")
    r_code = hex(red).split("x")[-1].rjust(2, "0")
    code = "#{0}{1}00".format(r_code, g_code)
    return code

# def agent_potrayal(agent):
#     potrayal = {
#     "Shape": "circle",
#     "Color": health_color(agent.health),
#     "Filled":"true",
#     "Layer": 0,
#     "r":0.5
#     }
#     return potrayal

def health_img(value):
    rounded_down = int(value / 10) * 10
    rounded_down = min(max(10, rounded_down), 100)
    return "res/monster/monster_small_{0}.png".format(rounded_down)

def agent_potrayal(agent):
    if isinstance(agent, SludgeMonster):
        potrayal = {
            "Shape": health_img(agent.health),
            "Layer": 1,
        }
    elif isinstance(agent, SludgeFood):
        potrayal = {
            "Shape":"circle",
            "Color":"#00FF00",
            "Layer": 0,
            "Filled":"true",
            "r": 0.5
        }
    return potrayal

EntropyController.set_entropy(1)
width=25
height=25
inital_pop = 25
grid = CanvasGrid(agent_potrayal, width, height, 500, 500)


chart1 = ChartModule([{"Label": "friendliness", "Color": "#AA0000"},
                            {"Label": "anger", "Color": "#222200"},
                            {"Label": "fertility", "Color": "#000022"},
                            {"Label": "movement_noise", "Color": "#00FF00"},
                            ],
                             data_collector_name="datacollector")

chart2 = ChartModule([{"Label": "leadership", "Color": "#FF00FF"},
                            {"Label": "follower_mult", "Color": "#00AA00"},
                            {"Label": "leader_attraction", "Color": "#00AAAA"},
                            {"Label": "food_attraction", "Color": "#22AA22"},
                            {"Label": "is_following", "Color": "#FFFF00"},
                             ],
                             data_collector_name="datacollector")

chart3= ChartModule([{"Label": "max_attack", "Color": "#002222"},
                            {"Label": "max_hug_benefit", "Color": "#AAAA00"},
                            {"Label": "_decay_mult", "Color": "#AA00AA"},
                             ],
                             data_collector_name="datacollector")

chart4 = ChartModule([{"Label": "sight", "Color": "#AA0000"},
                             ],
                             data_collector_name="datacollector")


server = ModularServer(SludgeMonsterModel,
                        [grid, chart1, chart2, chart3, chart4],
                        "SludgeMonsterModel",
                        {
                        "num_agents":inital_pop,
                        "width":width,
                        "height":height
                        })

