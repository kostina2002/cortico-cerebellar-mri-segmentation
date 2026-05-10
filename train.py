import torch
import torch.optim as optim
import gc
from models.cortico_cerebellar_unet import CorticoCerebellarUNet
from utils.losses import combined_loss
from utils.metrics import dice_score_per_volume

def train_epoch(model, loader, optimizer, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    total_cereb_loss = 0.0

    for slices, masks in loader:
        slices, masks = slices.to(device), masks.to(device)
        optimizer.zero_grad()

        out_masks, cereb_loss = model(slices, ablate=False,
                                      compute_cereb_loss=True,
                                      detach_cereb_input=True,
                                      current_epoch=epoch + 1)
        seg_loss = combined_loss(out_masks, masks)
        loss = seg_loss + cereb_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cereb_loss += cereb_loss.item()
        

    return total_loss / len(loader), total_cereb_loss / len(loader)

def validate(model, loader, device, ablate=False):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for slices, masks in loader:
            slices, masks = slices.to(device), masks.to(device)
            out_masks, _ = model(slices, ablate=ablate, compute_cereb_loss=False)
            loss = combined_loss(out_masks, masks)
            total_loss += loss.item()
            total_dice += dice_score_per_volume(out_masks, masks)

    return total_loss / len(loader), total_dice / len(loader)

def train_model(cfg, feedback_type, train_loader, val_loader, device, seed, lambda_fn=None):
    cfg.FEEDBACK_TYPE = feedback_type
    model = CorticoCerebellarUNet(cfg).to(device)

    if feedback_type == 'cerebellar':
        cerebellar_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'cerebellum' in name and param.requires_grad:
                cerebellar_params.append(param)
            elif param.requires_grad:
                other_params.append(param)

        optimizer = optim.Adam([
            {'params': cerebellar_params, 'lr': cfg.LEARNING_RATE * cfg.CEREBELLUM_LR_MULT},
            {'params': other_params, 'lr': cfg.LEARNING_RATE}
        ], lr=cfg.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'cereb_loss': [], 'lambda': []}
    best_dice = 0.0
    suffix = f"{feedback_type}_seed{seed}"

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, cereb_loss = train_epoch(model, train_loader, optimizer, device, epoch, cfg)
        val_loss, val_dice = validate(model, val_loader, device, ablate=False)
        scheduler.step(val_dice)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        if feedback_type == 'cerebellar' and lambda_fn is not None:
            current_lambda = lambda_fn(epoch + 1, cfg)
            history['cereb_loss'].append(cereb_loss)
            history['lambda'].append(current_lambda)
            
            print(f"Ep {epoch+1:02d} | {feedback_type:10} | λ={current_lambda:.3f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Cereb Loss: {cereb_loss:.4f} | Val Dice: {val_dice:.4f}")

        else:
            history['cereb_loss'].append(0.0)
            history['lambda'].append(0.0)
            print(f"Ep {epoch+1:02d} | {feedback_type:10} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Cereb Loss: 0.0000 | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f'{cfg.SAVE_PATH}/best_{suffix}.pth')
         

    return history, model
