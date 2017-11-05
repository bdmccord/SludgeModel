from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from agent_model import SludgeMonsterModel, EntropyController


def health_color(value):
    green = int(255 - ((100 - value) * 255/100.0))
    red = 255 - green

    g_code = hex(green).split("x")[-1].rjust(2, "0")
    r_code = hex(red).split("x")[-1].rjust(2, "0")
    code = "#{0}{1}00".format(r_code, g_code)
    return code

def agent_potrayal(agent):
    potrayal = {
    "Shape": "circle",
    "Color": health_color(agent.health),
    "Filled":"true",
    "Layer": 0,
    "r":0.5
    }
    return potrayal

EntropyController.set_entropy(1)
width=10
height=10
grid = CanvasGrid(agent_potrayal, width, height, 500, 500)
server = ModularServer(SludgeMonsterModel,
                        [grid],
                        "SludgeMonsterModel",
                        {
                        "num_agents":30,
                        "width":width,
                        "height":height
                        })

