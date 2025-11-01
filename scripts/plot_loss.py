"""
绘制训练损失曲线

从 training_log.json 读取数据并绘制训练/验证损失曲线。
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无头模式,适合服务器


def plot_loss_curves(log_path, output_dir="outputs"):
    """
    绘制训练和验证损失曲线
    
    Args:
        log_path: training_log.json 路径
        output_dir: 输出图片保存目录
    """
    # 读取日志
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    train_log = log_data.get('train', [])
    val_log = log_data.get('val', [])
    best_metric = log_data.get('best_metric', None)
    
    if not train_log:
        print("❌ 没有找到训练日志数据")
        return
    
    # 提取数据
    train_epochs = [i + 1 for i in range(len(train_log))]
    train_losses = [entry['loss'] for entry in train_log]
    train_accs = [entry.get('acc', None) for entry in train_log]

    val_epochs = []
    val_losses = []
    val_accs = []
    if val_log:
        for entry in val_log:
            # 从 val_log 中提取 epoch (假设按顺序记录)
            val_epochs.append(len(val_epochs) + 1)
            val_losses.append(entry['loss'])
            val_accs.append(entry.get('acc', None))
    
    # 创建图表（上：loss，下：accuracy）
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                         gridspec_kw={'height_ratios': [3, 2]})

    # Loss 曲线
    ax_loss.plot(train_epochs, train_losses,
                 label='Training Loss', marker='o', linewidth=2, markersize=4,
                 color='#1f77b4', alpha=0.9)
    if val_losses:
        ax_loss.plot(val_epochs, val_losses,
                     label='Validation Loss', marker='s', linewidth=2, markersize=4,
                     color='#ff7f0e', alpha=0.9)

        # 标记最佳验证损失
        if best_metric is not None and best_metric in val_losses:
            best_epoch = val_epochs[val_losses.index(best_metric)]
            ax_loss.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                            label=f'Best Val Loss (Epoch {best_epoch})')
            ax_loss.scatter([best_epoch], [best_metric], color='green', s=100, zorder=5, marker='*')

    ax_loss.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax_loss.legend(loc='best', fontsize=10)
    ax_loss.grid(True, alpha=0.3, linestyle='--')

    # Accuracy 曲线（仅当日志中存在 acc 字段）
    has_train_acc = any(a is not None for a in train_accs)
    has_val_acc = len(val_accs) > 0 and any(a is not None for a in val_accs)
    if has_train_acc:
        ax_acc.plot(train_epochs, [a if a is not None else float('nan') for a in train_accs],
                    label='Training Acc', marker='o', linewidth=2, markersize=4, color='#2ca02c')
    if has_val_acc:
        ax_acc.plot(val_epochs, [a if a is not None else float('nan') for a in val_accs],
                    label='Validation Acc', marker='s', linewidth=2, markersize=4, color='#d62728')

    ax_acc.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_acc.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax_acc.legend(loc='best', fontsize=10)
    ax_acc.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # 保存图片
    output_path = Path(output_dir) / 'loss_and_accuracy.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 损失/精度曲线已保存到: {output_path}")
    plt.close()
    
    # 打印统计信息
    print("\n📊 训练统计:")
    print(f"  总Epoch数: {len(train_log)}")
    print(f"  初始训练损失: {train_losses[0]:.4f}")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  训练损失降幅: {train_losses[0] - train_losses[-1]:.4f}")
    
    if val_losses:
        print(f"\n  初始验证损失: {val_losses[0]:.4f}")
        print(f"  最终验证损失: {val_losses[-1]:.4f}")
        if best_metric is not None:
            print(f"  最佳验证损失: {best_metric:.4f} (Epoch {val_epochs[val_losses.index(best_metric)]})")


def plot_detailed_analysis(log_path, output_dir="outputs"):
    """
    绘制详细分析图表(包括学习率、epoch时间等)
    
    Args:
        log_path: training_log.json 路径
        output_dir: 输出图片保存目录
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    train_log = log_data.get('train', [])
    val_log = log_data.get('val', [])
    
    if not train_log:
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Loss曲线 (左上)
    ax1 = axes[0, 0]
    train_epochs = list(range(1, len(train_log) + 1))
    train_losses = [entry['loss'] for entry in train_log]
    ax1.plot(train_epochs, train_losses, label='Training Loss', marker='o', markersize=3)
    
    if val_log:
        val_epochs = list(range(1, len(val_log) + 1))
        val_losses = [entry['loss'] for entry in val_log]
        ax1.plot(val_epochs, val_losses, label='Validation Loss', marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Epoch训练时间 (右上)
    ax2 = axes[0, 1]
    epoch_times = [entry.get('time', 0) for entry in train_log]
    ax2.plot(train_epochs, epoch_times, marker='o', color='purple', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time per Epoch')
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失变化率 (左下)
    ax3 = axes[1, 0]
    if len(train_losses) > 1:
        train_loss_deltas = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
        ax3.plot(range(2, len(train_log) + 1), train_loss_deltas, 
                marker='o', color='red', markersize=3, label='Train Loss Delta')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        if val_log and len(val_losses) > 1:
            val_loss_deltas = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
            ax3.plot(range(2, len(val_log) + 1), val_loss_deltas, 
                    marker='s', color='orange', markersize=3, label='Val Loss Delta')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Change')
        ax3.set_title('Loss Change per Epoch (negative = improvement)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 最佳模型标记 (右下)
    ax4 = axes[1, 1]
    if val_log:
        val_losses_full = [entry['loss'] for entry in val_log]
        improvements = [i for i in range(len(val_losses_full)) 
                       if i == 0 or val_losses_full[i] < min(val_losses_full[:i])]
        
        ax4.plot(val_epochs, val_losses_full, 'o-', label='Validation Loss', color='orange')
        for imp_idx in improvements:
            ax4.scatter([val_epochs[imp_idx]], [val_losses_full[imp_idx]], 
                       color='green', s=100, marker='*', zorder=5)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss')
        ax4.set_title('Validation Loss with Improvements (green stars)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_dir) / 'training_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 详细分析图已保存到: {output_path}")
    
    plt.close()


def find_log_file():
    """
    自动查找训练日志文件
    
    按优先级查找:
    1. /root/autodl-tmp/outputs/logs/training_log.json (服务器路径)
    2. logs/training_log.json (本地相对路径)
    3. outputs/logs/training_log.json (备用路径)
    """
    possible_paths = [
        Path("/root/autodl-tmp/outputs/logs/training_log.json"),  # 服务器路径
        Path("logs/training_log.json"),                          # 本地相对路径
        Path("outputs/logs/training_log.json"),                  # 备用路径
    ]
    
    for log_path in possible_paths:
        if log_path.exists():
            return log_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='绘制训练损失曲线')
    parser.add_argument('--log', type=str, default=None,
                       help='训练日志JSON文件路径 (默认: 自动查找)')
    parser.add_argument('--output', type=str, default='outputs',
                       help='输出图片保存目录 (默认: outputs)')
    parser.add_argument('--detailed', action='store_true',
                       help='生成详细分析图表')
    args = parser.parse_args()
    
    # 确定日志文件路径
    if args.log:
        log_path = Path(args.log)
    else:
        log_path = find_log_file()
        if log_path is None:
            print("❌ 找不到训练日志文件")
            print("   尝试的路径:")
            print("   - /root/autodl-tmp/outputs/logs/training_log.json")
            print("   - logs/training_log.json")
            print("   - outputs/logs/training_log.json")
            print("   请使用 --log 参数指定正确的路径")
            return
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_path}")
        print(f"   请确保训练已完成并生成了日志文件")
        return
    
    print(f"📖 读取日志: {log_path}")
    
    # 绘制基础损失曲线
    plot_loss_curves(log_path, args.output)
    
    # 绘制详细分析图
    if args.detailed:
        print("\n📊 生成详细分析图...")
        plot_detailed_analysis(log_path, args.output)
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
