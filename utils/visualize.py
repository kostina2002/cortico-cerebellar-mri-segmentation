import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_aggregated_curves(all_histories, save_dir, cfg):
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'none': 'blue', 'readout': 'green', 'cerebellar': 'orange'}
    epochs = range(1, cfg.NUM_EPOCHS + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparison of models with different feedback types', fontsize=16)
    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, alpha=0.3)

    # Dice plot
    ax = axes[0, 0]
    for fb_type, histories in all_histories.items():
        dice_matrix = np.array([h['val_dice'] for h in histories])
        mean_dice = np.mean(dice_matrix, axis=0)
        std_dice = np.std(dice_matrix, axis=0)
        ax.plot(epochs, mean_dice, 'o-', color=colors.get(fb_type, 'gray'),
                label=fb_type, linewidth=2, markersize=4)
        ax.fill_between(epochs, mean_dice - std_dice, mean_dice + std_dice,
                        color=colors.get(fb_type, 'gray'), alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Validation Dice')
    ax.legend()

    # Validation loss plot
    ax = axes[0, 1]
    for fb_type, histories in all_histories.items():
        loss_matrix = np.array([h['val_loss'] for h in histories])
        mean_loss = np.mean(loss_matrix, axis=0)
        std_loss = np.std(loss_matrix, axis=0)
        ax.plot(epochs, mean_loss, 'o-', color=colors.get(fb_type, 'gray'),
                label=fb_type, linewidth=2, markersize=4)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        color=colors.get(fb_type, 'gray'), alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()

    # Training loss plot
    ax = axes[0, 2]
    for fb_type, histories in all_histories.items():
        loss_matrix = np.array([h['train_loss'] for h in histories])
        mean_loss = np.mean(loss_matrix, axis=0)
        std_loss = np.std(loss_matrix, axis=0)
        ax.plot(epochs, mean_loss, 'o-', color=colors.get(fb_type, 'gray'),
                label=fb_type, linewidth=2, markersize=4)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        color=colors.get(fb_type, 'gray'), alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()

    # Cerebellar loss plot
    ax = axes[1, 0]
    if 'cerebellar' in all_histories:
        cereb_matrix = np.array([h.get('cereb_loss', [0]*cfg.NUM_EPOCHS) for h in all_histories['cerebellar']])
        mean_cereb = np.mean(cereb_matrix, axis=0)
        std_cereb = np.std(cereb_matrix, axis=0)
        ax.plot(epochs, mean_cereb, 'o-', color='orange', label='cerebellar', linewidth=2)
        ax.fill_between(epochs, mean_cereb - std_cereb, mean_cereb + std_cereb, color='orange', alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cerebellar loss')
    ax.set_title('Cerebellar prediction loss')
    ax.legend()

    # Boxplot of final Dice values
    ax = axes[1, 1]
    final_dice = {}
    for fb_type, histories in all_histories.items():
        final_dice[fb_type] = [h['val_dice'][-1] for h in histories]
    ax.boxplot(final_dice.values(), labels=final_dice.keys())
    ax.set_ylabel('Dice')
    ax.set_title('Final Dice distribution over 5 runs')

    # Bar plot of standard deviation of final Dice
    ax = axes[1, 2]
    std_vals = []
    labels = []
    for fb_type, histories in all_histories.items():
        dice_matrix = np.array([h['val_dice'] for h in histories])
        std_across_runs = np.std(dice_matrix, axis=0)
        labels.append(fb_type)
        std_vals.append(std_across_runs[-1])
    ax.bar(labels, std_vals, color=[colors.get(l, 'gray') for l in labels])
    ax.set_ylabel('Dice standard deviation')
    ax.set_title('Variability at last epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'aggregated_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Aggregated plots saved to {save_dir}")

def visualize_predictions_overview(models, val_loader, device, num_slices=4, save_path='overlay_predictions.png'):
    """Overlay predictions on original MRI slices."""
    models = {name: model.eval() for name, model in models.items()}
    with torch.no_grad():
        slices, masks = next(iter(val_loader))
        slices, masks = slices.to(device), masks.to(device)
        B, T, _, H, W = slices.shape
        idx_slice = T // 2
        slice_img = slices[0, idx_slice, 0].cpu().numpy()
        gt_mask = masks[0, idx_slice, 0].cpu().numpy()

        fig, axes = plt.subplots(1, len(models)+2, figsize=(4*(len(models)+2), 4))
        axes[0].imshow(slice_img, cmap='gray')
        axes[0].set_title('MRI original')
        axes[0].axis('off')
        axes[1].imshow(gt_mask, cmap='Reds', alpha=0.7)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        for i, (name, model) in enumerate(models.items()):
            pred, _ = model(slices[0:1, :, :, :, :])
            pred_mask = pred[0, idx_slice, 0].cpu().numpy() > 0.5
            axes[i+2].imshow(slice_img, cmap='gray')
            axes[i+2].imshow(pred_mask, cmap='Blues', alpha=0.7)
            axes[i+2].set_title(name)
            axes[i+2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()