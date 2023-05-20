import torch
from grad_june import Runner

from birds.models.model import Model


class June(Model):
    def __init__(self, config: dict, parameters_to_calibrate: list = ()):
        super().__init__()
        self.runner = Runner.from_parameters(config)
        self.parameters_to_calibrate = parameters_to_calibrate
        self.n_timesteps = config["timer"]["total_days"]

    def get_x(self):
        agent_data = self.runner.data["agent"]
        x = agent_data["transmission"]
        for key in ("susceptibility", "is_infected", "infection_time"):
            x = torch.vstack((x, agent_data[key]))
        for key in ("current_stage", "next_stage", "time_to_next_stage"):
            x = torch.vstack((x, agent_data["symptoms"][key]))
        return x

    def set_x(self, x):
        i = 0
        agent_data = self.runner.data["agent"]
        for key in ("transmission", "susceptibility", "is_infected", "infection_time"):
            agent_data[key] = x[-1, i, :]
            i += 1
        for key in ("current_stage", "next_stage", "time_to_next_stage"):
            agent_data["symptoms"][key] = x[-1, i, :]
            i += 1
        return x

    def trim_time_series(self, x):
        return x[-1:]

    def initialize(self, params):
        self.runner.timer.reset()
        self.runner.restore_initial_data()
        if "seed" in self.parameters_to_calibrate:
            self.runner.log_fraction_initial_cases = params[0]
        self.runner.set_initial_cases()
        x = self.get_x()
        return x.reshape(1, 7, -1)

    def set_parameters(self, params):
        for i, param in enumerate(self.parameters_to_calibrate):
            if "beta" in param:
                name = "_".join(param.split("_")[1:])
                self.runner.model.infection_networks.networks[name].log_beta = params[i]

    def step(self, params, x):
        self.set_parameters(params)
        self.set_x(x)
        next(self.runner.timer)
        self.runner.model(self.runner.data, self.runner.timer)
        x = self.get_x()
        return x.reshape(1, 7, -1)

    def observe(self, x):
        # return cumulative infections.
        return [x[:, 2, :].sum(1)]
