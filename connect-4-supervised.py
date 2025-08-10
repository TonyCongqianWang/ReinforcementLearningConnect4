#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from transformers import get_cosine_schedule_with_warmup
import itertools
from collections import defaultdict

os.makedirs('checkpoints', exist_ok=True)

# In[2]:


def argmax_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the count of correctly predicted samples for a batch based on
    matching the output argmax with *any* of the target argmax indices,
    returning the count as a tensor suitable for XLA accumulation.

    Args:
        output (torch.Tensor): Model output tensor of shape (n, w), where n is
                               batch size and w is the vector dimension.
        target (torch.Tensor): Target tensor of the same shape (n, w).

    Returns:
        torch.Tensor: A scalar tensor containing the count of correct predictions
                      for the batch, located on the same device as inputs.
                      Returns a zero tensor if batch size is 0.
    """
    if output.shape != target.shape:
        raise ValueError(f"Output and target tensors must have the same shape. Got {output.shape} and {target.shape}")
    if output.dim() != 2:
         raise ValueError(f"Output and target tensors must be 2D (n, w). Got {output.dim()} dimensions.")

    n, w = output.shape

    if n == 0:
        # Return a scalar tensor on the correct device
        return torch.tensor(0, dtype=torch.long, device=output.device) # Use long for counts

    # 0. Find the index of the maximum value in the output (common for both calculations)
    output_argmax = torch.argmax(output, dim=1)
    
    # --- Calculation for the original target ---

    # 1. Find the maximum value in the original target
    target_max_values = torch.max(target, dim=1, keepdim=True)[0]

    # 2. Create a boolean mask for the original target
    target_max_mask = (target == target_max_values)

    # 3. Check if the output_argmax index is True in the original target_max_mask
    # Ensure arange is on the same device as the tensors
    correct_predictions_bool_original = target_max_mask[torch.arange(n, device=output.device), output_argmax]

    # 4. Sum the boolean tensor to get the count of correct predictions for original target.
    correct_count_original = correct_predictions_bool_original.sum()

    # --- Calculation for the clipped target ---

    # 1. Clip the target values to (-1, 1)
    target_clipped = torch.clip(target, -1, 1)

    # 2. Find the maximum value in the clipped target
    target_clipped_max_values = torch.max(target_clipped, dim=1, keepdim=True)[0]

    # 3. Create a boolean mask for the clipped target
    target_clipped_max_mask = (target_clipped == target_clipped_max_values)

    # 4. Check if the output_argmax index is True in the clipped target_max_mask
    correct_predictions_bool_clipped = target_clipped_max_mask[torch.arange(n, device=output.device), output_argmax]

    # 5. Sum the boolean tensor to get the count of correct predictions for clipped target.
    correct_count_clipped = correct_predictions_bool_clipped.sum()

    return correct_count_original, correct_count_clipped


# In[3]:


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channel = max(channel // reduction, 1)
        self.fc_scale = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channel, channel, bias=True),
            nn.Sigmoid()
        )
        self.fc_offset = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channel, channel, bias=True),
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_scale = self.fc_scale(y).view(b, c, 1, 1)
        y_offset = self.fc_offset(y).view(b, c, 1, 1)
        return x * y_scale.expand_as(x) + y_offset.expand_as(x)

class InitialExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(InitialExtractor, self).__init__()
        layers = []
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=False))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        out = self.convs(x)
        return out

class Grouped1x1SumConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_groups,
            kernel_size=1,
            groups=num_groups,
            bias=True
        )

    def forward(self, x):
        B, _, H, W = x.shape
        G = self.num_groups
        Cout = self.out_channels

        out = self.conv(x)

        out = out.view(B, G, Cout, H, W)
        out = out.sum(dim=1)
        return out

class ConnectFourBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = (5, 5), mlp_factor=4, group_factor=2):
        super().__init__()

        ks_h, ks_w = kernel_size
        hidden_dim = in_channels * mlp_factor
        layers_0 = []

        layers_0.append(nn.Conv2d(in_channels, in_channels, kernel_size=(ks_h, ks_w),  padding=(ks_h // 2, ks_w // 2), groups=in_channels, bias=False))
        layers_0.append(LayerNorm2d(in_channels))

        layers_0.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
        layers_0.append(nn.SiLU(inplace=True))

        layers_0.append(nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False))

        self.conv_next = nn.Sequential(*layers_0)

        hidden_dim = in_channels * group_factor
        n_groups = 2 * group_factor
        ks_h, ks_w = 3, 3
        
        layers_1 = []
        layers_1.append(LayerNorm2d(in_channels))
        layers_1.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
        self.proj_up = nn.Sequential(*layers_1)

        layers_2 = []
        layers_2.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(ks_h, ks_w), padding=(ks_h // 2, ks_w // 2), groups=n_groups, bias=False))
        layers_2.append(Grouped1x1SumConv(hidden_dim, hidden_dim, num_groups=n_groups))
        layers_2.append(nn.SiLU(inplace=True))
        self.grouped_0 = nn.Sequential(*layers_2)

        layers_3 = []
        layers_3.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(ks_h, ks_w), padding=(ks_h // 2, ks_w // 2), groups=n_groups, bias=False))
        layers_3.append(Grouped1x1SumConv(hidden_dim, hidden_dim, num_groups=n_groups))
        layers_3.append(nn.SiLU(inplace=True))
        self.grouped_1 = nn.Sequential(*layers_3)

        layers_4 = []
        layers_4.append(SEBlock(hidden_dim))
        layers_4.append(nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False))
        self.proj_down = nn.Sequential(*layers_4)

    def forward(self, x):
        out = x
        summand = self.conv_next(x)
        out = out + summand

        up_proj = self.proj_up(out)
        summand = self.grouped_0(up_proj)
        up_proj = up_proj + summand
        summand = self.grouped_1(up_proj)
        up_proj = up_proj + summand

        summand = self.proj_down(up_proj)
        out = out + summand
        
        return out 


# In[4]:


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

class VisionTransformerWithTasks(nn.Module):
    def __init__(
        self,
        input_shape,
        num_layers,
        embed_dim,
        num_heads,
        mlp_dim,
        num_tasks=7 + 7 + 1,
    ):
        super().__init__()
        C, H, W = input_shape
        self.num_patches = H * W
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks

        self.proj = nn.Linear(C, embed_dim)
        self.task_tokens = nn.Parameter(torch.randn(1, num_tasks, embed_dim))
        self.pos_embed = nn.Embedding(self.num_patches + num_tasks, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Conv1d(
            in_channels=embed_dim * num_tasks,
            out_channels=num_tasks,
            kernel_size=1,
            groups=num_tasks
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.task_tokens, std=0.02)
        trunc_normal_(self.pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        task_tokens = self.task_tokens.expand(B, -1, -1)
        x = torch.cat((task_tokens, x), dim=1)

        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)

        x = self.transformer(x)
        task_outputs = x[:, :self.num_tasks]

        task_outputs = task_outputs.reshape(B, self.num_tasks * self.embed_dim, 1)
        logits = self.output_head(task_outputs).squeeze(-1)

        return logits


# In[5]:


class PolicyHead(nn.Module):
    def __init__(self, in_channels, input_height, input_width, expansion_factor_c=4, expansion_factor_w=4):
        super(PolicyHead, self).__init__()
        intermediate_channels = expansion_factor_c * in_channels
        self.conv_reduce_height_pw_0 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.conv_reduce_height_pw_1 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.conv_reduce_height_dw = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(input_height, 1), stride=(input_height, 1), groups=intermediate_channels, bias=False)
        self.conv_reduce_height_bn = nn.BatchNorm2d(intermediate_channels)
        self.silu_reduce_height_1 = nn.SiLU(inplace=True)
        self.silu_reduce_height_2 = nn.SiLU(inplace=True)
        self.conv_expand_pw = nn.Conv2d(intermediate_channels, in_channels, kernel_size=1, stride=1, bias=False)

        self.expansion_factor_w = expansion_factor_w
        intermediate_channels = expansion_factor_w * input_width
        self.conv_permuted_input_width_as_channels = nn.Conv2d(
            in_channels=input_width, 
            out_channels=intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.silu_permuted_input_width_as_channels = nn.SiLU(inplace=True)

        intermediate_channels = expansion_factor_w * in_channels
        self.final_logits_conv = nn.Conv2d(
            in_channels=intermediate_channels, 
            out_channels=1,
            kernel_size=1, 
            stride=1, 
            bias=True
        )
    
    def forward(self, x):
        out = self.conv_reduce_height_pw_0(x)
        out = self.conv_reduce_height_pw_1(out)
        out = self.silu_reduce_height_1(out)
        out = self.conv_reduce_height_dw(out)
        out = self.conv_reduce_height_bn(out)
        out = self.silu_reduce_height_2(out)
        out = self.conv_expand_pw(out)
        
        B, C2, H_one, W_orig = out.shape 
        
        permuted_for_conv = out.permute(0, 3, 2, 1).contiguous()
        
         # Shape: (B, W_orig, 1, C2)
        convolved_output = self.conv_permuted_input_width_as_channels(permuted_for_conv)
        convolved_output = self.silu_permuted_input_width_as_channels(convolved_output)
        
        # Shape: (B, k_expansion_factor * W_orig, 1, C2)
        current_tensor = convolved_output.permute(0, 3, 2, 1).contiguous()
        # Shape: (B, C2, 1, self.k_expansion_factor * W_orig)
        
        # --- Step 2: Rearrange to (B, k*C2, 1, W_orig) ---
        k = self.expansion_factor_w
        
        #(B, C2, 1, k * W_orig)
        temp_rearrange = current_tensor.squeeze(2) 
        #(B, C2, k * W_orig)
        
        temp_rearrange = temp_rearrange.view(B, C2, k, W_orig)
        #(B, C2, k, W_orig)
        
        temp_rearrange = temp_rearrange.permute(0, 2, 1, 3) 
        #(B, k, C2, W_orig)
        
        temp_rearrange = temp_rearrange.contiguous().view(B, k * C2, W_orig)
        #(B, k*C2, W_orig)
        
        tensor_for_final_conv = temp_rearrange.unsqueeze(2)
        #(B, k*C2, 1, W_orig)

        logits_intermediate = self.final_logits_conv(tensor_for_final_conv)
        # (B, 1, 1, W_orig)

        policy_logits = logits_intermediate.view(B, W_orig)
        # Shape: (B, W_orig)

        return policy_logits

class ValueHead(nn.Module):
    def __init__(self, in_channels, input_height, input_width, expansion_factor_c=4, expansion_factor_w=2):
        super(ValueHead, self).__init__()
        intermediate_channels = expansion_factor_c * in_channels
        self.conv_reduce_height_pw_0 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.conv_reduce_height_pw_1 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.conv_reduce_height_dw = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(input_height, 1), stride=(input_height, 1), groups=intermediate_channels, bias=False)
        self.conv_reduce_height_bn = nn.BatchNorm2d(intermediate_channels)
        self.silu_reduce_height_1 = nn.SiLU(inplace=True)
        self.silu_reduce_height_2 = nn.SiLU(inplace=True)
        self.conv_expand_pw = nn.Conv2d(intermediate_channels, in_channels, kernel_size=1, stride=1, bias=False)

        self.expansion_factor_w = expansion_factor_w
        intermediate_channels = expansion_factor_w * input_width
        self.conv_permuted_input_width_as_channels = nn.Conv2d(
            in_channels=input_width, 
            out_channels=intermediate_channels, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        self.silu_flattened_output = nn.SiLU(inplace=True)
        
        final_flattened_features = intermediate_channels * 1 * in_channels
        self.final_linear = nn.Linear(final_flattened_features, 1)

    
    def forward(self, x):
        out = self.conv_reduce_height_pw_0(x)
        out = self.conv_reduce_height_pw_1(out)
        out = self.silu_reduce_height_1(out)
        out = self.conv_reduce_height_dw(out)
        out = self.conv_reduce_height_bn(out)
        out = self.silu_reduce_height_2(out)
        out = self.conv_expand_pw(out)
        
        B, C2, H_one, W_orig = out.shape 
        
        permuted_for_conv = out.permute(0, 3, 2, 1).contiguous()
        
        # Shape: (B, W_orig, 1, C2)
        convolved_output = self.conv_permuted_input_width_as_channels(permuted_for_conv)
        flattened_output = convolved_output.view(B, -1) 
        silu_output = self.silu_flattened_output(flattened_output)
        scalar_value = self.final_linear(silu_output)
        
        return scalar_value


# In[6]:


input_channels = 2
input_height = 6
input_width = 7
output_width = 7

class Connect4ModelVit(nn.Module):
    def __init__(self):
        super().__init__()

        extractor_layers = []
        stage_0_width = 64
        num_conv_blocks = 5
        num_transformer_layers = 5
        
        transformer_args = {
            "input_shape" : (stage_0_width, input_height, input_width),
            "num_layers" : num_transformer_layers,
            "embed_dim" : 256,
            "num_heads" : 4,
            "mlp_dim" : 1024,
        }
        
        extractor_layers.append(
            InitialExtractor(in_channels=input_channels, out_channels=stage_0_width),
        )
        
        extractor_layers.extend([
            ConnectFourBlock(in_channels=stage_0_width,) for _ in range(num_conv_blocks)
        ])
        
        self.conv_extractor = nn.Sequential(*extractor_layers)

        self.transformer = VisionTransformerWithTasks(
            **transformer_args
        )

    def forward(self, x):
        convnet_features = self.conv_extractor(x)
        concatenated_output = self.transformer(convnet_features)

        policy_outputs, value_outputs, next_value_outputs \
            = torch.split(concatenated_output, [7, 1, 7], dim=-1)
        return policy_outputs, value_outputs, next_value_outputs
    
class Connect4ModelCvn(nn.Module):
    def __init__(self):
        super().__init__()

        extractor_layers = []
        stage_0_width = 64
        num_conv_blocks = 7
        
        extractor_layers.append(
            InitialExtractor(in_channels=input_channels, out_channels=stage_0_width),
        )
        
        extractor_layers.extend([
            ConnectFourBlock(in_channels=stage_0_width,) for _ in range(num_conv_blocks)
        ])
        
        self.conv_extractor = nn.Sequential(*extractor_layers)

        self.policy_logits_head = nn.Sequential(
                ConnectFourBlock(in_channels=stage_0_width,),
                PolicyHead(
                    in_channels=64, 
                    input_height=input_height,
                    input_width=input_width)
            )

        self.policy_regression_head = nn.Sequential(
                ConnectFourBlock(in_channels=stage_0_width,),
                PolicyHead(
                    in_channels=64, 
                    input_height=input_height,
                    input_width=input_width)
            )

        self.value_head = nn.Sequential(
                ConnectFourBlock(in_channels=stage_0_width,),
                ValueHead(
                    in_channels=64, 
                    input_height=input_height,
                    input_width=input_width)
            )

    def forward(self, x):
        convnet_features = self.conv_extractor(x)

        policy_outputs = self.policy_logits_head(convnet_features)
        value_outputs = self.value_head(convnet_features)
        next_value_outputs = self.policy_regression_head(convnet_features)
        return policy_outputs, value_outputs, next_value_outputs
    
def get_custom_model(model_type="vit_medium"):
    if model_type == "vit_medium":
        return Connect4ModelVit()
    if model_type == "cvn_tiny":
        return Connect4ModelCvn()
    raise Exception(f"unknown model_type: {model_type}")

def count_model_flops(model=None):
    from torch.utils.flop_counter import FlopCounterMode
    if model is None:
        model = get_custom_model()
    print("Model Architecture:")
    print(model)
    
    dummy_input = torch.randn(1, input_channels, input_height, input_width)
    
    with FlopCounterMode(model) as count:
        dummy_output = model(dummy_input)
        total_flops = count.get_total_flops()
    
    print(f"Total FLOPS: {total_flops}")
    print("Dummy Input Shape:", dummy_input.shape)
    print("Dummy Output:", dummy_output)


# In[7]:


from tqdm import tqdm

def move_to_device(batch, device):
        obs, targets = batch
        obs = obs.to(device, torch.float32)
        targets = targets.to(device, torch.float32)
        return obs, targets

mse_loss = nn.MSELoss(reduction='none')
kl_criterion = nn.KLDivLoss(reduction='none', log_target=True)

def calculate_loss(policy_outputs, value_outputs, next_value_outputs, labels):
    target_policy_log_probs = F.log_softmax(labels / 10, dim=-1)
    policy_log_probs = F.log_softmax(policy_outputs / 4, dim=-1)

    _, best_choice_indices = torch.max(torch.clamp(labels, min=-1, max=1), dim=-1)
    bad_choice_mask = torch.ones_like(policy_log_probs, dtype=torch.float)
    bad_choice_mask.scatter_(1, best_choice_indices.unsqueeze(1), 0)
    bad_choice_loss = - (policy_log_probs * bad_choice_mask).mean(dim=-1)

    policy_loss = kl_criterion(policy_log_probs, target_policy_log_probs).sum(dim=1)
    policy_loss = 0.8 * policy_loss + 0.002 * bad_choice_loss
    
    value_loss = mse_loss(value_outputs, torch.max(labels, dim=-1).values.unsqueeze(1)).mean(dim=1)
    next_value_loss = mse_loss(next_value_outputs, labels).mean(dim=1)

    combined_loss = 10000 * policy_loss + value_loss + 0.1 * next_value_loss

    loss_parts = {
        'combined': combined_loss,
        'policy': policy_loss,
        'value': value_loss,
        'next_value': next_value_loss
    }

    return combined_loss, loss_parts

def validate_model(model, val_loader, device, max_batches=None, use_tqdm=False, mine_hard=False):
    model.to(device)
    
    if mine_hard:
        hard_examples = []
        
    ema_loss = 0.0
    ema_alpha = 0.05
    
    # Validation loop
    model.eval()
    num_samples_in_val = 0
    val_loss = kl_loss_total = value_loss_total = next_value_loss_total = 0.0
    val_correct_count_strong = val_correct_count_weak = 0.0
    val_correct_count_strong_nv = val_correct_count_weak_nv = 0.0
    loader_slice =  itertools.islice(val_loader, max_batches) if max_batches else val_loader
    if use_tqdm:
        loader_slice = tqdm(loader_slice)
    with torch.no_grad():
        for batch in loader_slice:
            inputs, labels = move_to_device(batch, device)
            policy_outputs, value_outputs, next_value_outputs = model(inputs)

            loss, loss_parts = calculate_loss(policy_outputs, value_outputs, next_value_outputs, labels)
            
            batch_correct_count_strong, batch_correct_count_weak = argmax_accuracy(policy_outputs, labels)
            val_correct_count_strong += batch_correct_count_strong
            val_correct_count_weak += batch_correct_count_weak
            
            batch_correct_count_strong_nv, batch_correct_count_weak_nv = argmax_accuracy(next_value_outputs, labels)
            val_correct_count_strong_nv += batch_correct_count_strong_nv
            val_correct_count_weak_nv += batch_correct_count_weak_nv
            
            num_samples_in_batch = inputs.size(0)
            batch_loss_mean = loss.mean(dim=0).item()
            val_loss += batch_loss_mean * num_samples_in_batch
            kl_loss_total += loss_parts["policy"].sum(dim=0).item()
            value_loss_total += loss_parts["value"].sum(dim=0).item()
            next_value_loss_total += loss_parts["next_value"].sum(dim=0).item()
            num_samples_in_val += num_samples_in_batch
            
            if mine_hard:
                # Update the exponential moving average (EMA) of the loss
                if ema_loss == 0.0:
                    ema_loss = 0.8 * batch_loss_mean
                else:
                    ema_loss = ema_alpha * batch_loss_mean + (1 - ema_alpha) * ema_loss
                
                threshold = ema_loss * 10
                hard_example_indices = torch.where(loss > threshold)[0]
                
                if len(hard_example_indices) > 0:
                    hard_inputs = inputs[hard_example_indices]
                    hard_labels = labels[hard_example_indices]
                    hard_losses = loss[hard_example_indices]
                    hard_examples.append((hard_inputs, hard_labels, hard_losses))
            
    epoch_val_loss = {
            "combined": val_loss / num_samples_in_val, "policy": kl_loss_total / num_samples_in_val,
            "value": value_loss_total / num_samples_in_val, "next_value": next_value_loss_total / num_samples_in_val,
        }
    epoch_val_policy_acc = {
            "strong": val_correct_count_strong / num_samples_in_val, "weak": val_correct_count_weak / num_samples_in_val,
            "strong_nv": val_correct_count_strong_nv / num_samples_in_val, "weak_nv": val_correct_count_weak_nv / num_samples_in_val,
        }

    print(f"Validation Loss: {epoch_val_loss}")
    print(f"Validation Policy Acc: {epoch_val_policy_acc}")
    if mine_hard:
        if len(hard_examples) > 0:
            hard_inputs_tensor = torch.cat([ex[0] for ex in hard_examples], dim=0)
            hard_labels_tensor = torch.cat([ex[1] for ex in hard_examples], dim=0)
            hard_losses_tensor = torch.cat([ex[2] for ex in hard_examples], dim=0)
            return hard_inputs_tensor, hard_labels_tensor, hard_losses_tensor
        else:
            return None
            
    return epoch_val_policy_acc

def train_model(
    train_dataset,
    val_dataset,
    model: nn.Module,
    device,
    checkpoint_path: str = "checkpoint.pth",
    best_model_path: str = "best_model.pth",
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 10,
    num_batches_per_epoch = 10,
    warmup_fraction: float = 0.1,
    weight_decay: float = 0.01,
    use_tqdm=False,
):  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    total_steps = num_batches_per_epoch * epochs
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * warmup_fraction), total_steps)
        
    start_epoch = 0
    best_val_acc = 0
    model.to(device)
    
    epoch = start_epoch - 1
    step = -1
    model.train()
    continue_training = True
    
    # Training loop
    while continue_training:
        if use_tqdm:
            loader_subset = tqdm(train_loader)
        for batch in loader_subset:
            step += 1
            if step >= num_batches_per_epoch:
                step = 0
                running_loss_avg = {loss_name: loss_value / num_samples_in_epoch for loss_name, loss_value in running_loss.items()}
                print(f"Epoch [{epoch}/{epochs}] done, Training Loss: {running_loss_avg}")
                
                epoch_val_policy_acc = validate_model(model, val_loader, device, num_batches_per_epoch, use_tqdm=use_tqdm)
                # Save checkpoint and best model based on validation accuracy
                if epoch_val_policy_acc["strong"] > best_val_acc:
                    best_val_acc = epoch_val_policy_acc["strong"]
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Validation accuracy improved. Saved best model to {best_model_path}")
                torch.save(model.state_dict(), checkpoint_path)
                
            if step == 0:
                epoch += 1
                running_loss = defaultdict(float)
                num_samples_in_epoch = 0
                if epoch >= epochs:
                    continue_training = False
                    break
                print(f"Begin epoch {epoch} with LR: {optimizer.param_groups[0]['lr']:.6f}")
                
            inputs, labels = move_to_device(batch, device)

            optimizer.zero_grad()

            policy_outputs, value_outputs, next_value_outputs = model(inputs)

            combined_loss, loss_parts = calculate_loss(policy_outputs, value_outputs, next_value_outputs, labels)

            combined_loss.mean(dim=0).backward()
            optimizer.step()
            scheduler.step()

            num_samples_in_epoch += inputs.size(0)
            for loss_name, loss_value in loss_parts.items():
                running_loss[loss_name] += loss_value.sum(dim=0).item()
            if (step + 1) % math.ceil(num_batches_per_epoch / 10) == 0:
                running_loss_avg = {loss_name: loss_value / num_samples_in_epoch for loss_name, loss_value in running_loss.items()}
                print(f"Epoch [{epoch}/{epochs}], Step [{step} / {num_batches_per_epoch}], Running Loss: {running_loss_avg}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        

    print("Finished Training")
    print(f"Best validation acc achieved: {best_val_acc:.4f}")


# In[8]:


def load_dataset(dataset_path):
    import numpy as np
    import gc
    import torch

    try:
        print("Loading dataset with mmap_mode='r'...")

        dataset = np.load(dataset_path, mmap_mode='r')
        train_obs_np = dataset["x_train"]
        train_targets_np = dataset["y_train"]
        val_obs_np = dataset["x_val"]
        val_targets_np = dataset["y_val"]

        del dataset
        gc.collect() # Force garbage collection

        print("Converting NumPy arrays to PyTorch tensors with float16 precision (1/4)...")
        train_obs = torch.from_numpy(train_obs_np).to(dtype=torch.float16)
        del train_obs_np
        gc.collect()

        print("Converting NumPy arrays to PyTorch tensors with float16 precision (2/4)...")
        val_obs = torch.from_numpy(val_obs_np).to(dtype=torch.float16)
        del val_obs_np
        gc.collect()

        print("Converting NumPy arrays to PyTorch tensors with float16 precision (3/4)...")
        train_targets = torch.from_numpy(train_targets_np).to(dtype=torch.float16)
        del train_targets_np
        gc.collect()

        print("Converting NumPy arrays to PyTorch tensors with float16 precision (4/4)...")
        val_targets = torch.from_numpy(val_targets_np).to(dtype=torch.float16)
        del val_targets_np
        gc.collect()

        # Create TensorDatasets for use with PyTorch DataLoaders
        train_dataset = TensorDataset(train_obs, train_targets)
        val_dataset = TensorDataset(val_obs, val_targets)
        
        print("Dataset preparation complete.")
        return train_dataset, val_dataset
    except Exception as e:
        print(f"could not load data: {e}")
        return None


# In[9]:


import torch

def load_weights_only(filepath):
    """
    Loads only the weights (state_dict) from a .pth file.

    Args:
        filepath (str): The path to the .pth file.

    Returns:
        dict: The state dictionary containing the model weights, or None if an error occurs.
    """
    try:
        # Load the entire checkpoint
        checkpoint = torch.load(filepath, map_location="cpu")

        # Check if the loaded object is a state_dict directly
        if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            # Assume it's a state_dict if all keys are strings (common for weights)
            print(f"Successfully loaded state_dict from {filepath}")
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # If it's a dictionary containing a 'state_dict' key (common for full checkpoints)
            print(f"Successfully loaded 'state_dict' from checkpoint in {filepath}")
            state_dict = checkpoint['state_dict']
        else:
            print(f"Warning: The .pth file at {filepath} does not seem to contain a standard state_dict or a checkpoint with 'state_dict'.")
            print("Attempting to return the loaded object directly. Please inspect its content.")
            state_dict = checkpoint

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                # Remove the prefix and add the key-value pair to the new dict
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                # Keep keys that don't have the prefix as they are
                new_state_dict[key] = value
        return new_state_dict
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None


# In[10]:

import sys

arch = sys.argv[1]
num_run = sys.argv[2]

model = get_custom_model(arch)
count_model_flops(model)


# In[11]:


dataset_path = "./data/c4_data_enriched.npz"


# In[12]:


dataset_splits = load_dataset(dataset_path)
if dataset_splits is None:
    os._exit(1)

train_dataset, val_dataset = dataset_splits


# In[13]:


save_model_name = f"checkpoints/{arch}_r{num_run}"
checkpoint_file = f"{save_model_name}_checkpoint.pth"
best_model_file = f"{save_model_name}_best_model.pth"

load_file = f"{save_model_name}_starting_checkpoint.pth"

weights = load_weights_only(load_file)
if weights:
    print("Keys in loaded weights:", weights.keys())
    try:
        model.load_state_dict(weights)
        print("Successfully loaded weights into a new model.")
    except Exception as e:
        print(f"Could not load weights into a new model: {e}")


# In[14]:

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
model = torch.compile(model)

args = dict(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    model=model,
    batch_size=2 ** 11,
    learning_rate=8e-6,
    epochs=60,
    num_batches_per_epoch=4000,
    warmup_fraction=0.2,
    weight_decay=4e-5,
    checkpoint_path=checkpoint_file,
    best_model_path=best_model_file,
    device=device,
    use_tqdm=True
)


# In[15]:


train_model(**args)


# In[16]:


# Mine hard examples
if False:
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    hard_examples = validate_model(model, train_loader, device, max_batches=100, use_tqdm=True, mine_hard=True)
    if hard_examples:
        print(len(hard_examples[0]))
        hard_inputs, hard_targets, hard_losses = hard_examples
        hard_examples_tensor = {
            "obs": hard_inputs.cpu(),
            "targets": hard_targets.cpu(),
            "loss": hard_losses.cpu()
        }

        torch.save(hard_examples_tensor, "hard_examples.pt")


# In[17]:


del train_dataset
del val_dataset


# In[18]:


dataset = np.load(dataset_path, mmap_mode='r')
test_obs = torch.from_numpy(dataset["x_test"])
test_targets = torch.from_numpy(dataset["y_test"])

random_seed = 534984
indices = np.arange(len(test_obs))
np.random.seed(random_seed)
np.random.shuffle(indices)

test_obs = test_obs[indices]
test_targets = test_targets[indices]
test_dataset = TensorDataset(test_obs, test_targets)


# In[19]:


model.to(device)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
validate_model(model, test_loader, device, max_batches=None, use_tqdm=True)


# In[20]:


save_model = model
save_model.to(torch.device("cpu"))
save_model.eval()
save_model(torch.randn((1, 2, 6, 7)))

example_inputs = (torch.randn(1, 2, 6, 7),)
onnx_program = torch.onnx.export(save_model, example_inputs, dynamo=True)
onnx_program.save(f"{save_model_name}.onnx")
print(f"Model saved to {save_model_name}.onnx")


# In[21]:


import onnx

onnx_model = onnx.load(f"{save_model_name}.onnx")
onnx.checker.check_model(onnx_model)
import onnxruntime

example_inputs = (torch.randn(1, 2, 6, 7),torch.randn(1, 2, 6, 7),torch.randn(1, 2, 6, 7),)
onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
print(f"Input length: {len(onnx_inputs)}")
print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    f"./{save_model_name}.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
onnxruntime_outputs


# In[22]:


save_model(example_inputs[0])

