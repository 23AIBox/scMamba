import sys
sys.path.append(".")
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tensorboardX import SummaryWriter

from .utils.metrics import matching_metrics


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            checkpoint_dir: str,
            writer: SummaryWriter,
            device: str = 'cuda',
            init_epoch: int = 0,
        ) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_epoch = init_epoch
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer

    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.init_epoch, self.epochs + self.init_epoch):
            train_loss = self._train_epoch(train_loader, epoch)
            self._log_epoch(epoch, train_loss)

            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, epoch)
                self._log_epoch(epoch, val_loss)

            if epoch % 5 == 0:
                self._save_checkpoint(epoch, train_loss)

        self.writer.close()

    def _train_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        loss_all = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=False)

        for i, (rna, atac) in enumerate(loop):
            loss = self._train_step(rna, atac)
            running_loss += loss
            loss_all += loss

            if (i + 1) % 30 == 0:
                avg_loss = running_loss / 30
                self.writer.add_scalar("train loss (30 avg)", avg_loss, epoch * len(train_loader) + i)
                running_loss = 0.0

            loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
            loop.set_postfix(loss=loss)

        return loss_all / len(train_loader)

    def _train_step(self, rna: torch.Tensor, atac: torch.Tensor) -> float:
        rna, atac = rna.float().to(self.device), atac.float().to(self.device)
        self.optimizer.zero_grad()

        CausalLMOutput = self.model(rna, atac)
        rna_embeds, atac_embeds = CausalLMOutput.logits_omics_1, CausalLMOutput.logits_omics_2

        loss, _ = self.criterion(rna_embeds, atac_embeds)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _validate_epoch(self, val_loader, epoch: int) -> float:
        self.model.eval()
        running_loss = 0.0
        loop = tqdm(val_loader, total=len(val_loader), leave=False)

        for _, (rna, atac) in enumerate(loop):
            loss = self._val_step(rna, atac)
            running_loss += loss

            loop.set_description(f'Validating Epoch [{epoch}/{self.epochs}]')
            loop.set_postfix(loss=loss)

        return running_loss / len(val_loader)
    
    def _val_step(self, rna: torch.Tensor, atac: torch.Tensor) -> float:
        rna, atac = rna.float().to(self.device), atac.float().to(self.device)
        CausalLMOutput = self.model(rna, atac)
        rna_embeds, atac_embeds = CausalLMOutput.logits_omics_1, CausalLMOutput.logits_omics_2

        loss, _ = self.criterion(rna_embeds, atac_embeds)

        return loss.item()

    def _log_epoch(self, epoch: int, loss: float, is_train: bool=True) -> None:
        if is_train:
            self.writer.add_scalar("train loss", loss, epoch)
        else:
            self.writer.add_scalar("validate loss", loss, epoch)

    def _save_checkpoint(self, epoch: int, train_loss: float) -> None:
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoints/scMamba{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)


class Trainer_ddp:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            checkpoint_dir: str,
            writer: SummaryWriter,
            device: str = 'cuda',
            init_epoch: int = 0,
            use_ddp: bool = False,
        ) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_epoch = init_epoch
        self.checkpoint_dir = checkpoint_dir
        self.writer = writer
        self.use_ddp = use_ddp

    def fit(self, train_loader, val_loader=None):
        for epoch in range(self.init_epoch + 1, self.epochs + self.init_epoch + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            self._log_epoch(epoch, train_loss)

            if epoch % 5 == 0:
                self._save_checkpoint(epoch, train_loss)

        self.writer.close()

    def _train_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=False)

        for i, (rna, atac) in enumerate(loop):
            loss = self._train_step(rna, atac)
            running_loss += loss

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                self.writer.add_scalar("train loss (100 avg)", avg_loss, epoch * len(train_loader) + i)
                running_loss = 0.0

            loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
            loop.set_postfix(loss=loss)

        return loss.item()

    def _train_step(self, rna: torch.Tensor, atac: torch.Tensor) -> float:
        rna, atac = rna.float().to(self.device), atac.float().to(self.device)
        self.optimizer.zero_grad()

        CausalLMOutput = self.model(rna, atac, num_last_tokens=1)
        rna_embeds, atac_embeds = CausalLMOutput.logits_omics_1, CausalLMOutput.logits_omics_2

        loss, similarity = self.criterion(rna_embeds, atac_embeds)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # _validate and test methods can be added here...

    def _log_epoch(self, epoch: int, train_loss: float) -> None:
        self.writer.add_scalar("train loss", train_loss, epoch)

    def _save_checkpoint(self, epoch: int, train_loss: float) -> None:
        model_to_save = self.model.module if self.use_ddp else self.model
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoints/scMamba{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)
