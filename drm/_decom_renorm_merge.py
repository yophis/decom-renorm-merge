from typing import List, Union
import logging
import re

import torch
from torch import nn

from .utils import get_chained_attributes, get_inner_most_object_from_chained_attributes

torch.manual_seed(0)


__all__ = ["average", "disjoint_averaging", "drm_merging"]

NATIVE_STACKED_QKV_PARAM_REGEX = ".*in_proj.*"

MERGING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger('DRM')


def resolve_elementwise_sign_conflict(tensors, resolve_method="mass", dim=0, resolute_sign=None):
    """
    Returns tensors with conflicting sign entries dropped.
    resolve_method is either 'mass' or 'count'.
    """

    def resolve_zero_sign(signs, resolute_sign=None):
        if resolute_sign is not None:
            majority_sign = resolute_sign
        else:
            majority_sign = signs.sum().sign()
        signs[signs == 0] = majority_sign
        return signs

    if not isinstance(tensors, torch.Tensor):
        tensors = torch.stack(tensors, dim=dim)

    if resolve_method == "mass":
        sign_mass = tensors.sum(dim=dim, keepdim=True)
        signs = torch.sign(sign_mass)
    elif resolve_method == "count":
        sign_count = tensors.sign().sum(dim=dim, keepdim=True)
        signs = torch.sign(sign_count)
    else:
        raise NotImplementedError()

    signs = resolve_zero_sign(signs, resolute_sign=resolute_sign)
    sign_mask = tensors.sign() != signs
    return tensors.masked_fill(sign_mask, 0.0)


def disjoint_averaging(tensors, dim=0):
    """Returns mean with zeros ignored."""
    if not isinstance(tensors, torch.Tensor):
        tensors = torch.stack(tensors, dim=dim)

    divisor = torch.count_nonzero(tensors, dim=dim)
    divisor = torch.clamp(divisor, min=1.0)
    return tensors.sum(dim=dim) / divisor


def zero_out_non_topk(tensor, ratio):
    """Zeros out entries whose magnitudes aren't in top-`ratio`%."""
    if ratio == 1.0:
        return tensor

    original_shape = tensor.shape
    tensor = tensor.reshape(-1)

    k = len(tensor) - round(len(tensor) * ratio)
    kth_value = tensor.abs().kthvalue(k=k).values.item()  # smallest k-th.
    tensor.masked_fill_(tensor.abs() < kth_value, 0.0)
    return tensor.reshape(original_shape)


def zero_out_non_topk_ignore_zero(tensor, ratio):
    """Zeros out entries whose magnitudes aren't in top-`ratio`% of nonzero elements."""
    if ratio == 1.0:
        return tensor

    original_shape = tensor.shape
    tensor = tensor.view(-1)

    no_zero_tensor = tensor[tensor != 0.0]
    k = len(no_zero_tensor) - round(len(no_zero_tensor) * ratio)
    kth_value = no_zero_tensor.abs().kthvalue(k=k).values.item()

    tensor.masked_fill_(tensor.abs() < kth_value, 0.0)  # TODO: Optimize via using elem prod instead of masked fill
    return tensor.view(original_shape)


def average(tensor, disjoint: bool = False, dim: int = 0):
    if disjoint:
        return disjoint_averaging(tensor, dim=dim)
    return tensor.mean(dim=dim)


def right_drm(
    weights: List[torch.Tensor],
    coeffs,
    base_weight: torch.Tensor,
    basis_dropping_rate: float = 0.0,
    enable_disjoint_mean: bool = True,
):
    if basis_dropping_rate == 0.0:
        return weights

    if not isinstance(weights, torch.Tensor):
        weights = torch.stack(weights, dim=0)
    assert weights.dim() == 3

    num_models, n, m = weights.shape
    # Align into same column basis
    col_concat_weights = weights.permute((1, 0, 2)).reshape(n, -1)
    U, S, Vh = torch.linalg.svd(col_concat_weights, full_matrices=False)

    # rank_threshold = S.max() * max(n, m) * torch.finfo(S.dtype).eps
    # logging.debug(f'Rank threshold = {rank_threshold}, having highest svdval of {S.max().item()}.')
    rank_threshold = S.max().item() * 1e-2
    ranks = (S > rank_threshold).sum().item()

    # Trim out nullspace
    U = U[:, :ranks]
    S = S[:ranks]
    Vh = Vh[:ranks]

    # List of Vh matrix of each model [Vh1 Vh2 ... VhN]
    Vhs = list(torch.split(Vh, split_size_or_sections=[m] * num_models, dim=1))
    Ss = []
    for Vh in Vhs:
        Vh_norms = torch.linalg.vector_norm(Vh, ord=2, dim=1, keepdim=True)

        # Normalize right singular vectors
        Vh /= Vh_norms

        # Invert the norm value into singular values
        Ss.append(S * Vh_norms.squeeze(dim=1))

    Ss = torch.stack(Ss, dim=0)

    Vhs = torch.stack(Vhs, dim=0)
    SVhs = Ss.unsqueeze(-1) * zero_out_non_topk(Vhs, ratio=1 - basis_dropping_rate)
    SVhs = resolve_elementwise_sign_conflict(SVhs, resolve_method="mass", dim=0, resolute_sign=None)

    if not isinstance(coeffs, (int, float)):
        coeffs = torch.tensor(coeffs, device=SVhs.device, dtype=SVhs.dtype).view(-1, *([1] * (SVhs.ndim - 1)))
    SVhs = SVhs * coeffs
    SVh = average(SVhs, disjoint=enable_disjoint_mean, dim=0)
    return U @ SVh


def left_drm(
    weights: List[torch.Tensor],
    coeffs,
    base_weight: torch.Tensor,
    basis_dropping_rate: float = 0.0,
    enable_disjoint_mean: bool = True,
):
    if basis_dropping_rate == 0.0:
        return weights

    if not isinstance(weights, torch.Tensor):
        weights = torch.stack(weights, dim=0)
    assert weights.dim() == 3

    num_models, n, m = weights.shape
    # Align into same row basis
    row_concat_weights = weights.reshape(-1, m)
    U, S, Vh = torch.linalg.svd(row_concat_weights, full_matrices=False)

    # rank_threshold = S.max() * max(n, m) * torch.finfo(S.dtype).eps
    # logging.debug(f'Rank threshold = {rank_threshold}, having highest svdval of {S.max().item()}.')
    rank_threshold = S.max().item() * 1e-2
    ranks = (S > rank_threshold).sum().item()

    # Trim out nullspace
    U = U[:, :ranks]
    S = S[:ranks]  # (r,)
    Vh = Vh[:ranks]

    # List of U matrix of each model [U1 U2 ... UN]
    Us = list(torch.split(U, split_size_or_sections=[n] * num_models, dim=0))
    Ss = []
    for current_U in Us:
        U_norms = torch.linalg.vector_norm(current_U, ord=2, dim=0, keepdim=True)

        # Normalize right singular vectors
        current_U /= U_norms

        # Invert the norm value into singular values
        Ss.append(S * U_norms.squeeze(dim=0))

    Ss = torch.stack(Ss, dim=0)  # (num_models, r)

    Us = torch.stack(Us, dim=0)  # (num_models, n, r)
    UsS = Ss.unsqueeze(-2) * zero_out_non_topk(
        Us, ratio=1 - basis_dropping_rate
    )  # (num_models, 1, r) * (num_models, n, r)
    UsS = resolve_elementwise_sign_conflict(UsS, resolve_method="mass", dim=0, resolute_sign=None)

    if not isinstance(coeffs, (int, float)):
        coeffs = torch.tensor(coeffs, device=UsS.device, dtype=UsS.dtype).view(-1, *([1] * (UsS.ndim - 1)))
    UsS = UsS * coeffs
    US = average(UsS, disjoint=enable_disjoint_mean, dim=0)
    return US @ Vh


@torch.no_grad
def drm_merging(
    models: List[nn.Module],
    base_model: nn.Module,
    singular_matrices_drop_rate: float = 0.0,
    non_linear_weight_magnitude_dropping_rate: float = 0.0,
    enable_disjoint_mean: bool = True,
    enable_sign_resolution: bool = True,  # TODO: Check coverage
    merging_coeffs: Union[float, List[float]] = 1.0,
    ignore_patterns: List[str] = None,
    linear_weight_parameter_names: List[str] = None,
    is_horizontal_stack: bool = True,
    rank_based_stack_determination: bool = False,
    orthogonality_based_stack_determination: bool = False,
    weighted_orthogonality: bool = False,
    native_stacked_QKV: bool = False,
    dtype: str = "float32",
):
    """Decom-renorm-merge method for merging deep learning models."""

    def get_delta(parameter, base_parameter):
        return parameter - base_parameter

    def get_full_from_delta(delta, base_parameter):
        return delta + base_parameter

    def set_parameter(base_parameter: nn.Parameter, new_parameter_value: torch.Tensor):
        # Note: Assume `new_parameter_value` is delta instead of full weight
        new_parameter_value += base_parameter
        base_parameter.copy_(new_parameter_value)

    def detach_modules(model):
        """
        Temporarily detaches linear weight matrices from the model.
        This is to allow disjoint operation between linear weights and other parameters.
        """
        detached_parameters = {}
        # "module" here is alias for parameter
        for parameter_name, parameter in model.named_parameters():
            if parameter_name in linear_weight_parameter_names or any(
                re.match(pattern, parameter_name) for pattern in ignore_patterns
            ):
                detached_parameters[parameter_name] = parameter

                parameter_parent_module = get_inner_most_object_from_chained_attributes(model, parameter_name)
                base_name = parameter_name.rsplit(".", maxsplit=1)[-1]
                parameter_parent_module.register_parameter(base_name, None)

        return detached_parameters

    def attach_modules(model, detached_parameters):
        """Attaches back previously detached linear weights."""
        # "module" here is alias for parameter
        for detached_name in detached_parameters.keys():
            parameter = detached_parameters[detached_name]

            parameter_parent_module = get_inner_most_object_from_chained_attributes(model, detached_name)
            base_name = detached_name.rsplit(".", maxsplit=1)[-1]
            parameter_parent_module.register_parameter(base_name, parameter)

    def get_stacked_ranks(parameters):
        assert parameters.dim() == 3
        n_models, out_features, in_features = parameters.shape
        hstack_rank = torch.linalg.matrix_rank(parameters.transpose(0, 1).reshape(out_features, -1), rtol=0.1)
        vstack_rank = torch.linalg.matrix_rank(parameters.reshape(-1, in_features), rtol=0.1)
        return dict(hstack_rank=hstack_rank, vstack_rank=vstack_rank)

    def get_stacked_orthogonality(parameters, weighted: bool = False):
        def get_orthogonality(matrices: List[torch.Tensor]):
            flat_matrices = torch.stack([matrix.reshape(-1) for matrix in matrices], dim=0)
            inner_product = flat_matrices @ flat_matrices.transpose(0, 1)
            dense_orthogonality = inner_product - torch.eye(*inner_product.shape, out=torch.empty_like(inner_product))
            # return torch.linalg.matrix_norm(dense_orthogonality, ord='fro')
            return (1 - dense_orthogonality.abs()).sum()

        num_models, n, m = parameters.shape

        ## Align into same row basis
        row_concat_weights = parameters.reshape(-1, m)
        U, S, _ = torch.linalg.svd(row_concat_weights, full_matrices=False)

        rank_threshold = S.max().item() * 1e-2
        ranks = (S > rank_threshold).sum().item()
        U = U[:, :ranks]
        S = S[:ranks]
        Us = list(torch.split(U, split_size_or_sections=[n] * num_models, dim=0))
        if weighted:
            Us = [U @ torch.diag_embed(S) for U in Us]

        vstack_orthogonality = get_orthogonality(Us)

        ## Align into same column basis
        col_concat_weights = parameters.permute((1, 0, 2)).reshape(n, -1)
        _, S, Vh = torch.linalg.svd(col_concat_weights, full_matrices=False)

        rank_threshold = S.max().item() * 1e-2
        ranks = (S > rank_threshold).sum().item()
        Vh = Vh[:ranks]
        S = S[:ranks]
        Vhs = list(torch.split(Vh, split_size_or_sections=[m] * num_models, dim=1))
        if weighted:
            Vhs = [torch.diag_embed(S) @ Vh for Vh in Vhs]

        hstack_orthogonality = get_orthogonality(Vhs)

        return dict(
            hstack_orthogonality=hstack_orthogonality,
            vstack_orthogonality=vstack_orthogonality,
        )

    if orthogonality_based_stack_determination:
        logger.info("Orthogonality based stacking direction is enabled, overwriting `hstack` argument.")
    elif rank_based_stack_determination:
        logger.info("Rank based stacking direction is enabled, overwriting `hstack` argument.")

    if isinstance(merging_coeffs, (int, float)):
        merging_coeffs = [merging_coeffs] * len(models)
    assert len(merging_coeffs) == len(models)

    # Non-linear part
    # Trim non-linear weight parameters, such as bias terms or elementwise affine
    base_model_detached_modules = detach_modules(base_model)
    base_model_parameter_vector = nn.utils.parameters_to_vector(base_model.parameters())
    attach_modules(base_model, base_model_detached_modules)
    for model in models:
        detached_modules = detach_modules(model)
        parameter_vector = nn.utils.parameters_to_vector(model.parameters())
        deltas = get_delta(parameter=parameter_vector, base_parameter=base_model_parameter_vector)

        # Trim globally
        deltas = zero_out_non_topk(deltas, ratio=1 - non_linear_weight_magnitude_dropping_rate)

        # Set the trimmed delta back into the model, temporarily
        nn.utils.vector_to_parameters(deltas, model.parameters())

        attach_modules(model, detached_modules)

    for base_parameter_name, base_parameter in base_model.named_parameters():
        if any(re.match(pattern, base_parameter_name) for pattern in ignore_patterns):
            logger.info("Skipping parameter matching ignore pattern:", base_parameter_name)
            continue

        current_parameters = []
        for model in models:
            current_parameter = get_chained_attributes(model, base_parameter_name)
            # Apply get_delta if it's linear weight matrix,
            # otherwise the parameter is already delta
            if base_parameter_name in linear_weight_parameter_names:
                assert current_parameter.dim() == base_parameter.dim() == 2
                current_parameter = get_delta(parameter=current_parameter, base_parameter=base_parameter)
            current_parameters.append(current_parameter)
        current_parameters = torch.stack(current_parameters, dim=0)

        # Handle linear weight matrix
        if base_parameter_name in linear_weight_parameter_names and base_parameter.dim() == 2:
            # Perform basis merging  # TODO: REMOVE
            ## Dynamically determine stacking direction by orthogonality
            if orthogonality_based_stack_determination:
                orthogonalities = get_stacked_orthogonality(current_parameters, weighted=weighted_orthogonality)
                if orthogonalities["hstack_orthogonality"] > orthogonalities["vstack_orthogonality"]:
                    basis_merging_fn = right_drm
                    print("H-stack:", base_parameter_name)
                else:
                    basis_merging_fn = left_drm
                    print("V-stack:", base_parameter_name)
            ## Dynamically determine stacking direction by stacked rank
            elif rank_based_stack_determination:
                stacked_ranks = get_stacked_ranks(current_parameters)
                if stacked_ranks["hstack_rank"] < stacked_ranks["vstack_rank"]:
                    basis_merging_fn = right_drm
                    print("H-stack:", base_parameter_name)
                else:
                    basis_merging_fn = left_drm
                    print("V-stack:", base_parameter_name)
            ## Adhere to `hstack` argument
            else:
                basis_merging_fn = right_drm if is_horizontal_stack else left_drm

            # In open_clip's implementation of CLIP's ViT, the QKV projection matrices are stacked together
            # We split them and process each projection matrix separately if `native_stacked_QKV` is set to True
            if native_stacked_QKV and re.match(NATIVE_STACKED_QKV_PARAM_REGEX, base_parameter_name):
                split_QKV = torch.tensor_split(current_parameters, 3, dim=1)
                split_bases = torch.tensor_split(base_parameter, 3, dim=0)
                split_results = []
                for individual_QKV, individual_base in zip(split_QKV, split_bases):
                    individual_QKV = basis_merging_fn(
                        weights=individual_QKV.to(dtype=dtype, device=MERGING_DEVICE),
                        base_weight=individual_base,
                        basis_dropping_rate=singular_matrices_drop_rate,
                        enable_disjoint_mean=enable_disjoint_mean,
                        coeffs=merging_coeffs,
                    ).to(dtype=torch.float32, device="cpu")

                    split_results.append(individual_QKV)

                current_parameters = torch.cat(split_results, dim=0)
            else:
                current_parameters = basis_merging_fn(
                    weights=current_parameters.to(dtype=dtype, device=MERGING_DEVICE),
                    base_weight=base_parameter,
                    basis_dropping_rate=singular_matrices_drop_rate,
                    enable_disjoint_mean=enable_disjoint_mean,
                    coeffs=merging_coeffs,
                ).to(dtype=torch.float32, device="cpu")

            # If basis drop ratio is zero, basis merging above will be bypassed, as with sign resolve
            if singular_matrices_drop_rate == 0.0:
                current_parameters = average(current_parameters, disjoint=enable_disjoint_mean, dim=0)

        # Handle other parameters
        else:
            # Elect sign
            if enable_sign_resolution:
                current_parameters = resolve_elementwise_sign_conflict(
                    current_parameters, resolve_method="mass", resolute_sign=None
                )

            # Average
            _coeffs = torch.tensor(
                merging_coeffs,
                device=current_parameters.device,
                dtype=current_parameters.dtype,
            ).view(-1, *([1] * (current_parameters.ndim - 1)))
            current_parameters = current_parameters * _coeffs
            current_parameters = average(current_parameters, disjoint=enable_disjoint_mean, dim=0)

        set_parameter(new_parameter_value=current_parameters, base_parameter=base_parameter)

    return base_model
