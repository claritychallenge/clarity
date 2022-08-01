"""
Adopted from Asteroid
https://github.com/asteroid-team/asteroid/blob/master/asteroid/engine/system.py
"""
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class System(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):  # pylint: disable=W0613
        """Common forward step between training and validation.
        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            _batch_nb (int): The number of the batch in the epoch.
            _train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.
        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        inputs, targets = batch
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        return loss

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.
        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        loss = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        loss = self.common_step(batch, batch_nb, train=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics(
                {"hp_metric": hp_metric}, step=self.trainer.global_step
            )

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint
