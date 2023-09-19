from abc import ABC, abstractmethod
import logging
import numpy as np
import torch
from tqdm import trange
from typing import Callable, Union

logger = logging.getLogger("mcmc")

class MCMCKernel(ABC):

    """
    Abstract base class for MCMC kernels. These kernels specify how to sample the next step in
    a generic MCMC chain.

    **Arguments**

    - `prior`: The prior distribution. Must be differentiable in its argument.
    - `w`: The weight hyperparameter in generalised posterior.
    - `gradient_clipping_norm`: The norm to which the gradients are clipped.
    - `loss`: The loss function used in the exponent of the generalised likelihood term. Maps from data and chain state to loss.
    - `diff_mode`: The differentiation mode to use. Can be either 'reverse' or 'forward'.
    - `jacobian_chunk_size`: The number of rows computed at a time for the model Jacobian. Set to None to compute the full Jacobian at once.
    - `gradient_horizon`: The number of timesteps to use for the gradient horizon. Set 0 to use the full trajectory.
    - `device`: The device to use for training.
    """

    def __init__(
        self,
        prior: torch.distributions.Distribution,
        loss: Callable,
        w: float = 1.0,
        gradient_clipping_norm: float = np.inf,
        diff_mode: str = "reverse",
        jacobian_chunk_size: Union[int, None] = None,
        gradient_horizon: int = np.inf,
        device: str = "cpu",
        discretisation_method: str = "e-m",
    ):
        self.prior = prior
        self.w = w
        self.gradient_clipping_norm = gradient_clipping_norm
        self.loss = loss
        self.diff_mode = diff_mode
        self.jacobian_chunk_size = jacobian_chunk_size
        self.gradient_horizon = gradient_horizon
        self.device = device
        self._dim = self._verify_dim()

    def _verify_dim(self):
        """
        Checks the parameter dimension.
        """
        return self.prior.sample((1,)).shape[-1]

    @abstractmethod
    def step(
        self,
        current_state: torch.Tensor,
        data: torch.Tensor,
        *args,
        **kwargs
    ):
        pass
        

class MALA(MCMCKernel):
    """
    Class that generates a step in the chain of a Metropolis-Adjusted Langevin Algorithm run.

    **Arguments**

    - `prior`: The prior distribution. Must be differentiable in its argument.
    - `w`: The weight hyperparameter in generalised posterior.
    - `gradient_clipping_norm`: The norm to which the gradients are clipped.
    - `loss`: The loss function used in the exponent of the generalised likelihood term. Maps from data and chain state to loss.
    - `diff_mode`: The differentiation mode to use. Can be either 'reverse' or 'forward'.
    - `jacobian_chunk_size`: The number of rows computed at a time for the model Jacobian. Set to None to compute the full Jacobian at once.
    - `gradient_horizon`: The number of timesteps to use for the gradient horizon. Set 0 to use the full trajectory.
    - `device`: The device to use for training.
    - `discretisation_method`: How to discretise the overdamped Langevin diffusion. Default 'e-m' for Euler-Maruyama
    """

    def __init__(
        self,
        *args,
        discretisation_method: str = "e-m",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.discretisation_method = discretisation_method
        self._previous_log_density = None
        self._previous_grad_theta_of_log_density = None
        self._proposal = None

    def _compute_log_density_and_grad(self, state, data):
        _state = state.clone().detach()
        _state.requires_grad = True
        ell = self.loss(_state, data)
        log_prior_pdf = self.prior.log_prob(_state)
        log_density = -ell + log_prior_pdf * self.w
        log_density.backward()
        torch.nn.utils.clip_grad_norm_([_state], self.gradient_clipping_norm)
        return log_density.detach(), _state.grad

    def initialise_chain(self, state, data):
        log_density, grad_theta_of_log_density = self._compute_log_density_and_grad(
            state, data
        )
        self._previous_log_density = log_density
        self._previous_grad_theta_of_log_density = grad_theta_of_log_density
        self._proposal = None

    def step(
        self,
        current_state,
        data,
        scale: float = 1.0,
        covariance: Union[torch.Tensor, None] = None,
    ):
        """
        Returns a (torch.Tensor, bool) pair corresponding to (the current state of the chain, whether
        the current state resulted from an accept or reject decision in the Metropolis step).
        """

        if covariance is None:
            covariance = torch.eye(self._dim)
        sC = scale * covariance
        if self._previous_log_density is None:
            # This would happen if the user hasn't initialised the chain themselves
            self.initialise_chain(current_state, data)
        if self.discretisation_method == "e-m":
            if self._proposal is None:
                # This would happen if the user hasn't initialised the chain themselves
                gradient_term = torch.matmul(
                    sC, self._previous_grad_theta_of_log_density
                )
                mean = current_state + gradient_term
                logger.debug("Total mean =", mean)
                logger.debug("Gradient_term =", gradient_term)
                proposal = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean, covariance_matrix=2 * sC
                )
                self._proposal = proposal
            new_state = self._proposal.sample()
        else:
            raise NotImplementedError("Discretisation method not yet implemented")

        (
            new_log_density,
            grad_theta_of_new_log_density,
        ) = self._compute_log_density_and_grad(new_state, data)

        # Metropolis accept/reject step
        log_alpha = torch.log(torch.rand((1,))[0])
        # Compute reverse proposal logpdf
        if self.discretisation_method == "e-m":
            try:
                rev_proposal = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        new_state + torch.matmul(sC, grad_theta_of_new_log_density),
                        covariance_matrix=2 * sC,
                    )
                )
            except ValueError as e:
                logger.debug(new_state, grad_theta_of_new_log_density)
                raise e
        else:
            raise NotImplementedError("Discretisation method not yet implemented")
        log_accept_prob = (
            new_log_density
            + rev_proposal.log_prob(current_state)
            - self._previous_log_density
            - self._proposal.log_prob(new_state)
        )
        logger.debug("Current, new:", current_state, new_state)
        logger.debug(
            "Lalpha",
            log_accept_prob.item(),
            " = ",
            new_log_density.item(),
            "+",
            rev_proposal.log_prob(current_state).item(),
            "-",
            self._previous_log_density.item(),
            "-",
            self._proposal.log_prob(new_state).item(),
        )
        logger.debug("")
        accept = log_alpha < log_accept_prob
        if accept:
            self._previous_log_density = new_log_density
            self._previous_grad_theta_of_log_density = grad_theta_of_new_log_density
            self._proposal = rev_proposal
            return new_state, True
        return current_state, False


class MCMC:
    """
    Class that runs an MCMC chain.

    **Arguments**

    - `kernel`: An object with a .step() method that is used to generate the next sample in the chain.
    - `num_samples`: An integer specifying the number of samples to generate in the MCMC chain.
    - `progress_bar`: Whether to display a progress bar during training.
    - `progress_info`: Whether to display loss data during training.
    """

    def __init__(
        self,
        kernel: MCMCKernel,
        num_samples: int = 100_000,
        progress_bar: bool = True,
        progress_info: bool = True,
    ):
        self.kernel = kernel
        self.num_samples = num_samples
        self.progress_bar = progress_bar
        self.progress_info = progress_info
        # I suppose just in case something stops the program and you want to save the samples?
        self._samples = []

    def reset(self):
        self._samples = []

    def run(
        self,
        initial_state: torch.Tensor, 
        data: torch.Tensor, 
        *args, 
        seed: int = 0,
        T: int = 1, 
        **kwargs
    ):

        """
        Runs the MCMC chain.

        **Arguments**

        - `initial_state`: Starting location of the MCMC chain.
        - `data`: A torch.Tensor containing the data against which the simulator is compared.
        - `seed`: An integer specifying the initial random state of the RNG.
        - `T`: An integer specifying the number of steps between updates of the progress info (if shown).

        Additional arguments and keyword arguments can be passed, which will be passed to the kernel 
        .step() method.
        """

        assert isinstance(initial_state, torch.Tensor), "Initial state of the MCMC chain must be a torch.Tensor"
        assert isinstance(data, torch.Tensor), "The data must be passed as a torch.Tensor"

        if seed is not None:
            torch.manual_seed(seed)
        self.reset()

        if self.progress_bar:
            iterator = trange(self.num_samples)
        else:
            iterator = range(self.num_samples)

        self._samples.append(initial_state)
        state = initial_state
        if self.progress_info:
            total_accepted = 0
        for t in iterator:
            state, accept_step = self.kernel.step(state, data, *args, **kwargs)
            self._samples.append(state)
            if self.progress_info:
                if accept_step:
                    total_accepted += 1
                if t % T == 0:
                    iterator.set_postfix(
                        {"Acceptance rate": float(total_accepted) / (t + 1.0)}
                    )
        return self._samples
