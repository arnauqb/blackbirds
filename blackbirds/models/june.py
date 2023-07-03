import torch
from grad_june import Runner

from blackbirds.models.model import Model
from blackbirds.utils import soft_minimum


class June(Model):
    def __init__(self, config: dict, parameters_to_calibrate: list = ()):
        super().__init__()
        self.runner = Runner.from_parameters(config)
        self.parameters_to_calibrate = parameters_to_calibrate
        self.n_timesteps = config["timer"]["total_days"]

    def get_x(self):
        return None

    def set_x(self, x):
        return None

    def trim_time_series(self, x):
        return None

    def initialize(self, params):
        self.runner.timer.reset()
        self.runner.restore_initial_data()
        if "seed" in self.parameters_to_calibrate:
            seed = soft_minimum(torch.tensor(-0.1, device=params.device), params[0], 3)
            self.runner.log_fraction_initial_cases = seed
        self.runner.set_initial_cases()
        self.cases_per_timestep = self.runner.data["agent"].is_infected.sum().reshape(1)
        return None

    def set_parameters(self, params):
        for i, param in enumerate(self.parameters_to_calibrate):
            if "beta" in param:
                name = "_".join(param.split("_")[1:])
                self.runner.model.infection_networks.networks[name].log_beta = params[i]

    def step(self, params, x=None):
        self.set_parameters(params)
        next(self.runner.timer)
        self.runner.model(self.runner.data, self.runner.timer)
        cases = self.runner.data["agent"].is_infected.sum()
        self.cases_per_timestep = torch.vstack((self.cases_per_timestep, cases))
        return None

    def observe(self, x=None):
        return [self.cases_per_timestep[-1].reshape(1)]

    def run_and_observe(self, params):
        self.initialize(params)
        observed_outputs = self.observe()
        for _ in range(self.n_timesteps):
            self(params, None)
            observed_outputs = [
                torch.cat((observed_output, output))
                for observed_output, output in zip(observed_outputs, self.observe())
            ]
        return observed_outputs

    def detach_gradient_horizon(self, time_series, gradient_horizon):
        for prop in ("transmission", "susceptibility", "is_infected", "infection_time"):
            getattr(self.runner.data["agent"], prop).detach_()
        for prop in ("current_stage", "next_stage", "time_to_next_stage"):
            getattr(self.runner.data["agent"].symptoms, prop).detach_()
