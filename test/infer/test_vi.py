import torch
import pytest
import numpy as np
import normflows as nf

from blackbirds.models.random_walk import RandomWalk
from blackbirds.posterior_estimators import TrainableGaussian
from blackbirds.simulate import simulate_and_observe_model
from blackbirds.infer.vi import (
    VI,
    compute_regularisation_loss,
    compute_loss_and_jacobian_pathwise,
    compute_and_differentiate_loss,
)


class TestDifferentiateLoss:
    def test__compute_loss_pathwise(self, mock_estimator, mock_loss):
        observed_outputs = [4 * torch.ones(3, 2)]
        parameters, loss, jacobians = compute_loss_and_jacobian_pathwise(
            mock_loss, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, 2.0 * torch.ones(2))
        assert loss == torch.tensor(0.0)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([0.0, 0.0]))

        observed_outputs = [
            torch.tensor([2.0, 3.0, 4.0]).reshape(-1, 1) * torch.ones(3, 2)
        ]
        parameters, loss, jacobians = compute_loss_and_jacobian_pathwise(
            mock_loss, mock_estimator, 5, observed_outputs
        )
        assert len(parameters) == 5
        for param in parameters:
            assert torch.allclose(param, torch.tensor([2.0, 2.0]))
        assert loss == torch.tensor(5 / 3)
        assert len(jacobians) == 5
        for jacob in jacobians:
            assert torch.allclose(jacob, torch.tensor([4.0, 4.0]))

    @pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
    @pytest.mark.parametrize("gradient_estimation_method", ["score", "pathwise"])
    def test__compute_loss(
        self, mock_estimator, mock_loss, diff_mode, gradient_estimation_method
    ):
        observed_outputs = [4 * torch.ones(3, 2)]
        loss = compute_and_differentiate_loss(
            mock_loss,
            mock_estimator,
            5,
            observed_outputs,
            diff_mode=diff_mode,
            gradient_estimation_method=gradient_estimation_method,
        )
        assert np.isclose(loss, 0.0)
        observed_outputs = [
            torch.tensor([2.0, 3.0, 4.0]).reshape(-1, 1) * torch.ones(3, 2)
        ]
        loss = compute_and_differentiate_loss(
            mock_loss,
            mock_estimator,
            5,
            observed_outputs,
            diff_mode=diff_mode,
            gradient_estimation_method=gradient_estimation_method,
        )
        assert np.isclose(loss, 5 / 3)
        if gradient_estimation_method == "pathwise":
            # for score since the mock one is constant this is 0...
            assert np.isclose(mock_estimator.p.grad.item(), 16, rtol=1e-3)


class TestRegularisation:
    def test_regularisation(self):
        n_samples = 100000
        # define two normal distributions
        dist1 = TrainableGaussian([0.0], 1)
        dist2 = TrainableGaussian([0.0], 1)
        # check that the KL divergence is 0
        assert np.isclose(
            compute_regularisation_loss(dist1, dist2, n_samples).detach(), 0.0
        )
        # define two normal distributions with different means
        dist1 = TrainableGaussian([0.0], 1)
        dist2 = TrainableGaussian([1.0], 1)
        # check that the KL divergence is the right result
        assert np.isclose(
            compute_regularisation_loss(dist1, dist2, n_samples).detach(),
            0.5,
            rtol=1e-2,
        )


class TestVI:
    @pytest.fixture(name="loss")
    def make_loss(self):
        class Loss:
            def __init__(self):
                self.model = RandomWalk(100)
                self.loss_fn = torch.nn.MSELoss()

            def __call__(self, params, data):
                observed_outputs = simulate_and_observe_model(self.model, params, 0)
                return self.loss_fn(observed_outputs[0], data[0])

        return Loss()

    def test_iniitalize_to_prior(self, make_flow, loss):
        prior = torch.distributions.Normal(0.0, 1.0)
        data = loss.model.run_and_observe(torch.logit(torch.tensor(([0.2]))))
        posterior_estimator = make_flow(1)
        optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-6)
        vi = VI(
            loss=loss,
            posterior_estimator=posterior_estimator,
            prior=prior,
            data=data,
            optimizer=optimizer,
            initialize_estimator_to_prior=True,
            initialization_lr=1e-3,
            w=100.0,
            progress_bar=False,
            n_samples_per_epoch=1,
            n_samples_regularisation=1000,
        )
        vi.run(1, max_epochs_without_improvement=100)
        kd_loss = compute_regularisation_loss(posterior_estimator, prior, 10000)
        assert np.isclose(kd_loss.item(), 0.0, atol=0.1)

    @pytest.mark.parametrize("diff_mode", ["forward", "reverse"])
    def test_random_walk(self, diff_mode, loss):
        """
        Tests inference in a random walk model.
        """
        true_ps = torch.logit(torch.tensor([0.25]))  # , 0.5, 0.75]
        prior = torch.distributions.Normal(0.0, 1.0)
        for true_p in true_ps:
            data = loss.model.run_and_observe(torch.tensor([true_p]))
            posterior_estimator = TrainableGaussian(
                torch.logit(torch.tensor([0.4])).numpy(), 0.1
            )
            posterior_estimator.sigma.requires_grad = False
            optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=1e-2)
            vi = VI(
                loss=loss,
                posterior_estimator=posterior_estimator,
                prior=prior,
                data=data,
                optimizer=optimizer,
                diff_mode=diff_mode,
                w=0.0,
                progress_bar=False,
                n_samples_per_epoch=1,
            )
            vi.run(100, max_epochs_without_improvement=100)
            ## check correct result is within 2 sigma
            assert np.isclose(posterior_estimator.mu.item(), true_p, atol=0.1)

    def test__train_regularisation_only(self, loss):
        data = loss.model.run_and_observe(torch.tensor([0.5]))

        prior = torch.distributions.Normal(3.0, 1)

        posterior_estimator = TrainableGaussian([0.0], 1.0)
        posterior_estimator.sigma.requires_grad = False

        optimizer = torch.optim.Adam(posterior_estimator.parameters(), lr=5e-2)
        vi = VI(
            loss=loss,
            posterior_estimator=posterior_estimator,
            prior=prior,
            data=data,
            optimizer=optimizer,
            n_samples_per_epoch=1,
            w=10000.0,
            progress_bar=False,
            n_samples_regularisation=1000,
        )
        vi.run(100, max_epochs_without_improvement=np.inf)
        posterior_estimator.load_state_dict(vi.best_estimator_state_dict)
        assert np.isclose(posterior_estimator.mu.item(), 3, rtol=0.1)
        assert np.isclose(posterior_estimator.sigma.item(), 1, rtol=0.1)
