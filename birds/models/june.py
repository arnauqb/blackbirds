import torch
from grad_june import Runner

from birds.models.model import Model
from birds.utils import soft_minimum


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
            seed = soft_minimum(torch.tensor(-0.1, device=params.device), params[0], 3)
            self.runner.log_fraction_initial_cases = seed
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
        return [x[:,2,:].sum(1)]
        #cases_by_age = self.get_cases_by_age(x).reshape(1,-1)
        #return [cases_by_age[:,i] for i in range(self.runner.age_bins.shape[0]-1)]

    def get_cases_by_age(self, x):
        data = self.runner.data
        age_bins = self.runner.age_bins
        ret = torch.zeros(age_bins.shape[0] - 1, device=x.device)
        for i in range(1, age_bins.shape[0]):
            age_bin = age_bins[i]
            mask1 = data["agent"].age < age_bins[i]
            mask2 = data["agent"].age > age_bins[i - 1]
            mask = mask1 * mask2
            ret[i - 1] = (data["agent"].is_infected * mask).sum() / self.runner.population_by_age[age_bin.item()]
        return ret
