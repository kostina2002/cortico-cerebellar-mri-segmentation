import os
import gc
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import cfg
from dataset import AtriaSegDataset
from train import train_model
from models.cortico_cerebellar_unet import CorticoCerebellarUNet
from utils.visualize import plot_aggregated_curves, visualize_predictions_overview

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(run_multiple_seeds=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("CORTICO-CEREBELLAR SEGMENTATION (with slice subsampling)")
    print("="*60)

    if run_multiple_seeds:
        SEEDS = [42, 123, 456, 789, 101112]
    else:
        SEEDS = [42]

    feedback_types = ['readout', 'cerebellar']
    all_histories = {fb: [] for fb in feedback_types}
    last_models = {}

    cfg.create_dirs()

    for seed in SEEDS:
        print("\n" + "="*50)
        print(f"RUN WITH SEED = {seed}")
        print("="*50)
        set_seed(seed)

        train_ds = AtriaSegDataset(
            cfg.DATA_PATH, split='train',
            target_size=cfg.INPUT_SIZE,
            max_slices=cfg.MAX_SLICES_PER_VOLUME,
            slice_step=cfg.SLICE_STEP,
            random_state=seed
        )
        val_ds = AtriaSegDataset(
            cfg.DATA_PATH, split='val',
            target_size=cfg.INPUT_SIZE,
            max_slices=cfg.MAX_SLICES_PER_VOLUME,
            slice_step=cfg.SLICE_STEP,
            random_state=seed
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                                shuffle=False, num_workers=cfg.NUM_WORKERS)

        for fb in feedback_types:
            print("\n" + "-"*40)
            print(f"Training with feedback = {fb}, seed = {seed}")
            print("-"*40)
            hist, model = train_model(cfg, fb, train_loader, val_loader, device, seed)
            all_histories[fb].append(hist)
            if seed == SEEDS[-1]:
                last_models[fb] = model
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Save histories
    with open(os.path.join(cfg.HISTORY_DIR, 'all_histories.pkl'), 'wb') as f:
        pickle.dump(all_histories, f)

    plot_aggregated_curves(all_histories, cfg.PLOTS_DIR, cfg)

    # Visualize predictions using the last seed's best models
    val_ds_last = AtriaSegDataset(
        cfg.DATA_PATH, split='val',
        target_size=cfg.INPUT_SIZE,
        max_slices=cfg.MAX_SLICES_PER_VOLUME,
        slice_step=cfg.SLICE_STEP,
        random_state=SEEDS[-1]
    )
    val_loader_last = DataLoader(val_ds_last, batch_size=cfg.BATCH_SIZE,
                                 shuffle=False, num_workers=cfg.NUM_WORKERS)

    best_models = {}
    for fb in feedback_types:
        original_feedback = cfg.FEEDBACK_TYPE
        cfg.FEEDBACK_TYPE = fb
        model = CorticoCerebellarUNet(cfg).to(device)
        cfg.FEEDBACK_TYPE = original_feedback
        ckpt = f'{cfg.SAVE_PATH}/best_{fb}_seed{SEEDS[-1]}.pth'
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            best_models[fb] = model
            print(f"Loaded best model for {fb} (seed {SEEDS[-1]})")
        else:
            print(f"Checkpoint {ckpt} not found, using last saved model.")
            best_models[fb] = last_models.get(fb)

    visualize_predictions_overview(best_models, val_loader_last, device,
                                   num_slices=4,
                                   save_path=os.path.join(cfg.PLOTS_DIR, 'overlay_predictions.png'))

    print(f"\nAll experiments completed. Results saved in {cfg.PLOTS_DIR}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(run_multiple_seeds=True, device=device)