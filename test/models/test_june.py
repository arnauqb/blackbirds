import yaml
import torch

from grad_june.paths import default_config_path
from blackbirds.models.june import June


class TestJune:
    def test__run(self):
        config = yaml.safe_load(open(default_config_path))
        model = June(config=config, parameters_to_calibrate=("seed", "beta_household"))
        assert model.parameters_to_calibrate == ("seed", "beta_household")
        assert model.n_timesteps == 15
        x = model.run_and_observe(torch.tensor([-1.0, 0.1]))
        assert len(x) == 1
        assert x[0].shape == (15 + 1,)

        parameters = torch.tensor([-1.0, 1.0])

        def loss_f(parameters):
            x = model.run_and_observe(parameters)
            return x[0][-1]

        jac_f = torch.func.jacfwd(loss_f, argnums=0, randomness="same")
        jacobian = jac_f(parameters)
        assert jacobian[0] != 0
        assert jacobian[1] != 0
