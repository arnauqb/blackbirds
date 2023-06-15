from abc import ABC, abstractmethod

import torch


class Model(ABC):
    @abstractmethod
    def initialize(self, params):
        pass

    @abstractmethod
    def step(self, params, x):
        pass

    def __call__(self, params, x):
        return self.step(params, x)

    @abstractmethod
    def observe(self, x):
        pass

    @abstractmethod
    def trim_time_series(self, time_series):
        pass

    def run(self, params):
        x = self.initialize(params)
        for _ in range(self.n_timesteps):
            x_t = self.step(params, x)
            x = torch.vstack((x, x_t))
        return x

    def run_and_observe(self, params):
        time_series = self.initialize(params)
        observed_outputs = self.observe(time_series)
        for _ in range(self.n_timesteps):
            time_series = self.trim_time_series(
                time_series
            )  # gets past time-steps needed to compute the next one.
            x = self(params, time_series)
            observed_outputs = [
                torch.cat((observed_output, output))
                for observed_output, output in zip(observed_outputs, self.observe(x))
            ]
            time_series = torch.cat((time_series, x))
        return observed_outputs

    def detach_gradient_horizon(self, time_series, gradient_horizon):
        """
        Detaches time-steps previous to `gradient_horizon`.
        """
        # detach past time-steps from the computational graph
        no_gradient_time_series = time_series[
            : len(time_series) - gradient_horizon
        ].detach()
        # stack with the last gradient_horizon time-steps
        time_series = torch.cat(
            (
                no_gradient_time_series,
                time_series[len(time_series) - gradient_horizon :],
            )
        )
        return time_series
