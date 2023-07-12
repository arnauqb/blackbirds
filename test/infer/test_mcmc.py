import torch
import pytest
import numpy as np

from blackbirds.models.random_walk import RandomWalk
from blackbirds.infer import mcmc
from blackbirds.simulate import simulate_and_observe_model


class TestMALA:
 
    def test_normal(self):
        """
        Tests inference on conjugate univariate Normal example.
        """

        # Create data
        torch.manual_seed(0)
        sigma = 1.
        normal_data = torch.distributions.normal.Normal(-1., sigma).sample((3,))

        # Create prior
        mu_0, sigma_0 = 1., 2.
        normal_prior = torch.distributions.normal.Normal(mu_0, sigma_0)

        # Create negative log-likelihood function
        def negative_log_likelihood(theta, data):
            dist = torch.distributions.normal.Normal(theta, sigma)
            return - dist.log_prob(data).sum()

        # Perform trial run
        mala = mcmc.MALA(normal_prior, negative_log_likelihood, w=1.)
        sampler = mcmc.MCMC(mala, 10_000)
        trial_samples = sampler.run(torch.tensor([-.5]), normal_data)
        thinned_trial_samples = torch.stack(trial_samples)[::10].T
        scale = torch.cov(thinned_trial_samples)

        # Perform proper run using trial run to inform MCMC hyperparameters
        mala = mcmc.MALA(normal_prior, negative_log_likelihood, 1.)
        sampler = mcmc.MCMC(mala, 20_000)
        post_samples = sampler.run(
            torch.tensor([thinned_trial_samples.mean()]), normal_data, scale=scale
        )
        thinned_post_samples = post_samples[::20]

        # Compare to true posterior – estimate KL divergence
        factor = (1/sigma_0**2 + normal_data.size/sigma**2)
        true_mean = (mu_0/sigma_0**2 + normal_data.sum()/sigma**2)/factor
        true_std  = 1./torch.sqrt(factor)
        obtained_mean = thinned_post_samples.mean()
        obtained_std  = thinned_post_samples.std()
        ratio_stds = obtained_std/true_std
        assert np.isclose(0.5 * ( 
            (ratio_stds)**2 
            - 1 
            + ((true_mean - obtained_mean)/true_std)**2 
            + 2 * torch.log(1./ratio_stds)
        ), 0.)

    def test_multivariate_normal(self, mv_normal_data):
        """
        Tests inference on conjugate bivariate Normal example.
        """

        # Create data
        torch.manual_seed(0)
        sigma = torch.tensor([[1., 0.4,], [0.4, 2.]])
        true_mean = torch.tensor([-1., 2.])
        true_density = torch.distributions.multivariate_normal.MultivariateNormal(true_mean, sigma)
        normal_data = true_density.sample((3,))

        # Create prior
        mu_0, sigma_0 = torch.tensor([2., 0.]), torch.tensor([[2., 0.,], [0., 1.]])
        normal_prior = torch.distributions.multivariate_normal.MultivariateNormal(mu_0, sigma_0)

        # Create negative log-likelihood function
        def negative_log_likelihood(theta, data):
            dist = torch.distributions.multivariate_normal.MultivariateNormal(theta, sigma)
            return - dist.log_prob(data).sum()

        # Perform trial run
        mala = mcmc.MALA(normal_prior, negative_log_likelihood, w=1.)
        sampler = mcmc.MCMC(mala, 10_000)
        trial_samples = sampler.run(mu_0, normal_data)
        thinned_trial_samples = torch.stack(trial_samples)[::10].T
        cov = torch.cov(thinned_trial_samples)
        init_state = thinned_trial_samples.mean(dim=1)

        # Perform proper run using trial run to inform MCMC hyperparameters
        mala = mcmc.MALA(normal_prior, negative_log_likelihood, w=1.)
        sampler = mcmc.MCMC(mala, 20_000)
        post_samples = sampler.run(init_state, normal_data, covariance=cov)
        thinned_post_samples = post_samples[::20]

        # Compare to true posterior – estimate KL divergence
        inv_sigma_0 = torch.inverse(sigma_0)
        inv_sigma = torch.inverse(sigma)
        inv_additions = torch.inverse(inv_sigma_0 + data_size*inv_sigma)
        true_mean = torch.matmul(inv_additions,
                                 (torch.matmul(inv_sigma_0, mu_0) + data_size*torch.matmul(inv_sigma, data.mean(dim=0))))
        true_inv_cov  = inv_sigma_0 + data_size*inv_sigma
        obtained_mean = thinned_post_samples.mean(dim=0)
        obtained_cov  = torch.cov(thinned_post_samples.T)
        diff_means = true_mean - obtained_mean
        kl = 0.5 * (
            torch.trace(
                torch.matmul(true_inv_cov, obtained_cov))
                - 2 
                + torch.matmul(diff_means, 
                               torch.matmul(true_inv_cov, diff_means)
                       )
                - torch.logdet(obtained_cov)
                - torch.logdet(true_inv_cov)
        )

        assert np.isclose(kl, 0.)

    def test_random_walk(self):

        """
        Test MALA on discrete random walk example.
        """

        torch.manual_seed(0)
        # Setup model and real data
        rw = RandomWalk(n_timesteps=100) 
        true_p = torch.logit(torch.tensor(0.25))
        true_data = rw.observe(rw.run(torch.tensor([true_p])))[0]
        prior = torch.distributions.Normal(true_p + 0.2, 1)

        # Define negative log likelihood
        def negative_log_likelihood(logit_p, y):
            p = torch.sigmoid(logit_p)
            log1mp = torch.log(1. - p)
            T = y.shape[0]
            ll = (T - 1) * log1mp
            logp = torch.log(p)
            ll += ((y[1:] - y[:-1] + 1)/ 2.).sum() * (logp - log1mp)
            return -ll

        # Trial run of MCMC
        mala = mcmc.MALA(prior, negative_log_likelihood, 
                 w=1., gradient_clipping_norm=np.inf)
        sampler = mcmc.MCMC(mala, 10_000)
        trial_samples = sampler.run(
            true_data, (true_p + 0.2).unsqueeze(0), scale=1.
        )
        thinned_trial_samples = torch.stack(trial_samples)[::10].T
        cov = torch.cov(thinned_trial_samples)

        # Proper run
        mala = mcmc.MALA(prior, negative_log_likelihood,
                         w=1., gradient_clipping_norm=np.inf)
        sampler = mcmc.MCMC(mala, 20_000)
        post_samples = sampler.run(true_data, thinned_trial_samples.mean(dim=1),
                                   covariance=cov.unsqueeze(0).unsqueeze(0))

        # Large number of data points so _should_ get good agreement between
        # mean and MLE (by Bernstein-von-Mises theorem?)
        est_p = torch.logit( ((true_data[1:] - true_data[:-1]).mean() + 1.)/2. )
        assert np.isclose(post_samples.mean(), est_p)
