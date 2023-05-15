def compute_regularisation_loss(posterior_estimator, prior, n_samples):
    r"""Estimates the KL divergence between the posterior and the prior using n_samples through Monte Carlo.
    The formula is:

    $$
    \mathbb{E}_{q(z|x)}[\log q(z|x) - \log p(z)] \approx \frac{1}{N} \sum_{i=1}^N \log q(z_i|x) - \log p(z_i)
    $$

    Arguments:
        posterior_estimator (torch.distributions.Distribution): The posterior distribution.
        prior (torch.distributions.Distribution): The prior distribution.
        n_samples (int): The number of samples to use for the Monte Carlo estimate.
    Example:
        >>> import torch
        >>> from birds.regularisation import compute_regularisation
        >>> # define two normal distributions
        >>> dist1 = torch.distributions.Normal(0, 1)
        >>> dist2 = torch.distributions.Normal(0, 1)
        tensor(0.)
        >>> dist1 = torch.distributions.Normal(0, 1)
        >>> dist2 = torch.distributions.Normal(1, 1)
        tensor(0.5)
    """
    # sample from the posterior
    z = posterior_estimator.sample((n_samples,))
    # compute the log probability of the samples under the posterior
    log_prob_posterior = posterior_estimator.log_prob(z)
    # compute the log probability of the samples under the prior
    log_prob_prior = prior.log_prob(z)
    # compute the Monte Carlo estimate of the KL divergence
    kl_divergence = (log_prob_posterior - log_prob_prior).mean()
    return kl_divergence
