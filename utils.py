import torch


def get_memory_allocated(batch_idx, check_step, train_step='optimizer.step()'):
    if batch_idx % check_step == 0:
        print(f"Batch {batch_idx}, Memory allocated after {train_step}: {torch.cuda.memory_allocated()} bytes")


def get_layer_gradients(model, epoch, batch_idx, check_step, get_mean_grad=False):
    if batch_idx % check_step == 0:
        # Get gradients of each layer
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"Epoch {epoch}, Batch {batch_idx}, Layer {i} - Gradient Norm: {grad_norm:.6f}")
        # Get average norm gradient over all layers
        if get_mean_grad:
            grad_norms_list = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
            if grad_norms_list:
                print(f"Epoch {epoch}, Batch {batch_idx} - Average Gradient Norm: {sum(grad_norms_list) / len(grad_norms_list):.6f}")