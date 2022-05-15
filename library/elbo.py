# torch imports
import torch


class ELBO_distribution:

    def __init__(self, pdf_distribution=torch.distributions.laplace.Laplace):
        self.pdf_distribution = pdf_distribution

    def __call__(self, x, outputs, lambda_KL=0.01, eta_KL=0.01):

        """
        :param x:
        :param outputs:
        :param lambda_KL:
        :param eta_KL:
        :return: ELBO loss

        """
        mu_x = outputs['mu_x']
        b_x = outputs['b_x']
        mu_z = outputs["mu_z"]
        sigma_z = outputs["sigma_z"]
        mu_c = outputs["mu_c"]
        sigma_c = outputs["sigma_c"]

        # Initialize Laplace with given parameters.
        pdf = self.pdf_distribution(mu_x, b_x)
        # Calculate mean of likelihood over T and x-dimension.
        likelihood = pdf.log_prob(x).mean(dim=1).mean(dim=1)

        # Calculate KL-divergence of p(c) and q(z)
        v_z = sigma_z**2  # Variance of q(z)
        v_c = sigma_c**2  # Variance of p(c)
        kl_z = -0.5 * torch.sum(1 + torch.log(v_z) - mu_z**2 - v_z, dim=2)
        kl_c = -0.5 * torch.sum(1 + torch.log(v_c) - mu_c**2 - v_c, dim=2)

        kl_c = torch.sum(kl_c, dim=1)  # Sum over the T dimension.
        kl_z = kl_z[:, 0]  # Get rid of extra x_dim dimension.

        # Calculate the Evidence Lower Bound (ELBO)
        ELBO = -likelihood + lambda_KL * (kl_z + eta_KL * kl_c)

        # Return mean over all examples in batch
        return torch.mean(ELBO, dim=0)
