import torch
from torch import nn

__all__ = ['minimal_model_averaging', 'task_arithmetic_merging']

def minimal_model_averaging(models, base_model):
    parameter_vectors = []
    for model in models:
        model_parameter_vector = nn.utils.parameters_to_vector(model.parameters())
        parameter_vectors.append(model_parameter_vector)

    parameter_vector = torch.stack(parameter_vectors, dim=0).mean(dim=0)
    nn.utils.vector_to_parameters(parameter_vector, base_model.parameters())
    return base_model

def task_arithmetic_merging(models, base_model, coeffs):
    base_parameter_vector = nn.utils.parameters_to_vector(base_model.parameters())
    parameter_vectors = []
    for model in models:
        model_parameter_vector = nn.utils.parameters_to_vector(model.parameters())
        model_parameter_vector = model_parameter_vector - base_parameter_vector
        parameter_vectors.append(model_parameter_vector)

    if coeffs is None:
        parameter_vector = torch.stack(parameter_vectors, dim=0).mean(dim=0)
    else:
        if isinstance(coeffs, (int, float)):
            coeffs = [coeffs] * len(models)
        assert len(coeffs) == len(parameter_vectors)
        
        parameter_vector = torch.stack(parameter_vectors, dim=0)
        if not isinstance(coeffs, (int, float)):
            coeffs = torch.tensor(coeffs, device=parameter_vector.device, dtype=parameter_vector.dtype).view(-1, *([1] * (parameter_vector.ndim-1)))
        
        parameter_vector = parameter_vector * coeffs
        parameter_vector = parameter_vector.sum(dim=0)
    nn.utils.vector_to_parameters(parameter_vector+base_parameter_vector, base_model.parameters())
    return base_model