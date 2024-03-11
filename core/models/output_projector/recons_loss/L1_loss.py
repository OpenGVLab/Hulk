import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    """
    L1 loss for mean and variance of source and target features.
    Used for regularization among tokens of different modalities.
    :param loss_weight_mean: weight for mean difference
    :param loss_weight_var: weight for variance difference
    """
    def __init__(self, loss_weight_mean=1., loss_weight_var=1.):
        super(L1Loss, self).__init__()
        self.loss_weight_mean = loss_weight_mean
        self.loss_weight_var = loss_weight_var

    def forward(self, source_features, target_features):
        """
        :param source_features: input token features from source modality
        :param target_features: input token features from target modality
        :return: loss dict: {'loss_mean_diff': mean_diff, 'loss_var_diff': var_diff}
        """

        # source_features, target_features: [bs, token_length, feature_dims]
        # calculate mean and variance of source and target features
        source_mean = source_features.mean(dim=[0,1])
        source_var = source_features.var(dim=[0,1], unbiased=False)
        target_mean = target_features.mean(dim=[0,1])
        target_var = target_features.var(dim=[0,1], unbiased=False)

        # calculate the distance between source and target mean and variance
        mean_diff = torch.abs(source_mean - target_mean).mean()
        var_diff = torch.abs(source_var - target_var).mean()

        # calculate regularization loss as the sum of mean and variance differences
        # loss = mean_diff + var_diff
        return {'loss_mean_diff': mean_diff*self.loss_weight_mean, 'loss_var_diff': var_diff*self.loss_weight_var}