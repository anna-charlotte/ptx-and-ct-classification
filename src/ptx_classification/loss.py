import torch
import torch.nn as nn

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


class FocalWithLogitsLoss(nn.Module):
    """
    Criterion that computes Focal loss for multilabel problem.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N, C)` where each value is 0.0 or 1.0.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction)


def binary_focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    assert (
        input.size() == target.size()
    ), f"input.size() = {input.size()}, target.size() = {target.size()}"

    input_soft = torch.sigmoid(input)
    p_t = torch.mul(target, input_soft) + torch.mul((1 - target), (1 - input_soft))
    loss = alpha * (-1) * torch.pow(1.0 - p_t, gamma) * torch.log(p_t)

    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Function that computes focal loss for multilabel problem.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data (logits) tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Returns:
        the computed loss.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> focal_loss(logits, labels, **kwargs)
        tensor(21.8725)
    """
    num_classes = input.size()[-1]
    loss = torch.stack(
        [
            binary_focal_loss(
                input=input[:, i : i + 1],
                target=target[:, i : i + 1],
                alpha=alpha,
                gamma=gamma,
                reduction=reduction,
            )
            for i in range(num_classes)
        ],
        dim=1 if reduction == "none" else 0,
    )

    return torch.sum(loss)
