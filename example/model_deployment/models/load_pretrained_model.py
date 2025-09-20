#!/usr/bin/env python3
"""
加载预训练模型并转换为我们的Tensor格式
"""

import torch
import torchvision
import numpy as np
import os

def load_pretrained_resnet18():
    """加载预训练的ResNet-18模型"""
    print("正在加载预训练ResNet-18模型...")
    
    # 加载预训练模型
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    
    print(f"模型加载成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def save_weights_to_binary(model, output_dir):
    """将模型权重保存为二进制文件"""
    print("正在保存模型权重...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各层权重
    weights_info = {}
    
    # 卷积层权重
    conv1_weight = model.conv1.weight.data.numpy()
    
    # 保存为二进制文件
    conv1_weight.tofile(os.path.join(output_dir, 'conv1_weight.bin'))
    
    # 检查是否有偏置
    if model.conv1.bias is not None:
        conv1_bias = model.conv1.bias.data.numpy()
        conv1_bias.tofile(os.path.join(output_dir, 'conv1_bias.bin'))
        weights_info['conv1'] = {
            'weight_shape': conv1_weight.shape,
            'bias_shape': conv1_bias.shape
        }
        print(f"conv1权重形状: {conv1_weight.shape}")
        print(f"conv1偏置形状: {conv1_bias.shape}")
    else:
        # 创建零偏置
        conv1_bias = np.zeros(conv1_weight.shape[0], dtype=np.float32)
        conv1_bias.tofile(os.path.join(output_dir, 'conv1_bias.bin'))
        weights_info['conv1'] = {
            'weight_shape': conv1_weight.shape,
            'bias_shape': conv1_bias.shape
        }
        print(f"conv1权重形状: {conv1_weight.shape}")
        print(f"conv1偏置形状: {conv1_bias.shape} (零偏置)")
    
    # 全连接层权重
    fc_weight = model.fc.weight.data.numpy()
    
    fc_weight.tofile(os.path.join(output_dir, 'fc_weight.bin'))
    
    # 检查是否有偏置
    if model.fc.bias is not None:
        fc_bias = model.fc.bias.data.numpy()
        fc_bias.tofile(os.path.join(output_dir, 'fc_bias.bin'))
        weights_info['fc'] = {
            'weight_shape': fc_weight.shape,
            'bias_shape': fc_bias.shape
        }
        print(f"fc权重形状: {fc_weight.shape}")
        print(f"fc偏置形状: {fc_bias.shape}")
    else:
        # 创建零偏置
        fc_bias = np.zeros(fc_weight.shape[0], dtype=np.float32)
        fc_bias.tofile(os.path.join(output_dir, 'fc_bias.bin'))
        weights_info['fc'] = {
            'weight_shape': fc_weight.shape,
            'bias_shape': fc_bias.shape
        }
        print(f"fc权重形状: {fc_weight.shape}")
        print(f"fc偏置形状: {fc_bias.shape} (零偏置)")
    
    # 保存权重信息
    with open(os.path.join(output_dir, 'weights_info.txt'), 'w') as f:
        f.write("ResNet-18 预训练权重信息\n")
        f.write("=" * 40 + "\n")
        f.write(f"conv1权重形状: {conv1_weight.shape}\n")
        f.write(f"conv1偏置形状: {conv1_bias.shape}\n")
        f.write(f"fc权重形状: {fc_weight.shape}\n")
        f.write(f"fc偏置形状: {fc_bias.shape}\n")
        f.write(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}\n")
    
    print(f"权重已保存到: {output_dir}")
    return weights_info

def main():
    """主函数"""
    print("=== 预训练模型加载工具 ===")
    
    # 加载预训练模型
    model = load_pretrained_resnet18()
    
    # 保存权重
    output_dir = "pretrained_weights"
    weights_info = save_weights_to_binary(model, output_dir)
    
    print("\n=== 权重信息 ===")
    for layer_name, info in weights_info.items():
        print(f"{layer_name}:")
        print(f"  权重形状: {info['weight_shape']}")
        print(f"  偏置形状: {info['bias_shape']}")
    
    print("\n预训练模型加载完成！")

if __name__ == "__main__":
    main()
 