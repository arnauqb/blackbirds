import torch
import pytest
import numpy as np
import normflows as nf

from birds.models.random_walk import RandomWalk
from birds.calibrator import Calibrator 
from birds.regularisation import compute_regularisation_loss
from birds.posterior_estimators import TrainableGaussian

def make_flow(n_parameters):
    K = 4
    torch.manual_seed(0)
    flows = []
    for i in range(K):
        flows.append(nf.flows.MaskedAffineAutoregressive(n_parameters, 20, num_blocks=2))
    q0 = nf.distributions.DiagGaussian(n_parameters)
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    return nfm

class TestCalibrator:
    def test_iniitalize_to_prior(self):
        rw = RandomWalk(100)
        true_ps = [0.25]  # , 0.5, 0.75]
        prior = torch.distributions.Normal(0.0, 1.0)
        data = rw.run_and_observe(torch.tensor([0.2]))
        posterior_estimator = make_flow(1)
        optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-6) 
        calib = Calibrator(
            model=rw,
            posterior_estimator=posterior_estimator,
            prior=prior,
            data=data,
            optimizer=optimizer,
            initialize_flow_to_prior=True,
            initialization_lr=1e-3,
            w=100.0,
            progress_bar=False,
            n_samples_per_epoch=1,
        )
        calib.run(1, max_epochs_without_improvement=100)
        kd_loss = compute_regularisation_loss(posterior_estimator, prior, 10000)
        assert np.isclose(kd_loss.item(), 0.0, atol=0.1)

    @pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
    def test_random_walk(self, diff_mode):
        """
        Tests inference in a random walk model.
        """
        rw = RandomWalk(100)
        true_ps = [0.25]  # , 0.5, 0.75]
        prior = torch.distributions.Normal(0.0, 1.0)
        for true_p in true_ps:
            data = rw.run_and_observe(torch.tensor([true_p]))
            posterior_estimator = TrainableGaussian([0.4], 0.1)
            posterior_estimator.sigma.requires_grad = False
            optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=5e-3)
            calib = Calibrator(
                model=rw,
                posterior_estimator=posterior_estimator,
                prior=prior,
                data=data,
                optimizer=optimizer,
                diff_mode=diff_mode,
                w=0.0,
                progress_bar=False,
                n_samples_per_epoch=5,
            )
            calib.run(25, max_epochs_without_improvement=100)
            ## check correct result is within 2 sigma
            assert np.isclose(posterior_estimator.mu.item(), true_p, atol = 0.1)

    def test__train_regularisation_only(self):
        rw = RandomWalk(2)
        data = rw.run_and_observe(torch.tensor([0.5]))

        prior = torch.distributions.Normal(3.0, 1)

        posterior_estimator = TrainableGaussian([0.0], 1.0)
        posterior_estimator.sigma.requires_grad = False

        optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=5e-2)
        calib = Calibrator(
            model=rw,
            posterior_estimator=posterior_estimator,
            prior=prior,
            data=data,
            optimizer=optimizer,
            n_samples_per_epoch=1,
            w=10000.0,
            progress_bar=False,
        )
        calib.run(100, max_epochs_without_improvement=np.inf)
        posterior_estimator.load_state_dict(calib.best_model_state_dict)
        assert np.isclose(posterior_estimator.mu.item(), 3, rtol=0.1)
        assert np.isclose(posterior_estimator.sigma.item(), 1, rtol=0.1)
