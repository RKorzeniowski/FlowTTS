import torch


def return_const(epoch):
    return 1


class ChainedAnnCyclicScheduler:
    def __init__(self, optimizer, lr, steps_per_epoch, epochs, **kwargs):
        self.scheduler_change_step = steps_per_epoch * epochs * 3 / 4
        self.lr_scheduler_ann = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_change_step,
            eta_min=lr / 100
        )
        self.lr_scheduler_cyc = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr / 100,
            max_lr=5 * lr / 100,
            step_size_up=2,
        )
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.scheduler_change_step:
            self.lr_scheduler_ann.step()
        else:
            self.lr_scheduler_cyc.step()


def get_constant_scheduler(optimizer, lr, **kwargs):
    """use defined function that returns constant so that Experiment class can
    be pickled"""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=return_const)


def get_cos_ann_scheduler(optimizer, lr, steps_per_epoch, epochs, **kwargs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps_per_epoch * epochs,
        eta_min=lr / 10
    )


def get_on_plateau_scheduler(optimizer, **kwargs):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


def get_ann_warm_restarts_scheduler(optimizer, lr, steps_per_epoch, epochs, **kwargs):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=epochs * steps_per_epoch // 7,
        T_mult=2,
        eta_min=lr / 4,
        last_epoch=-1,
    )


def get_one_cycle_scheduler(optimizer, lr, steps_per_epoch, epochs, **kwargs):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )


def get_cos_ann_into_cyc_scheduler(optimizer, lr, steps_per_epoch, epochs, **kwargs):
    return ChainedAnnCyclicScheduler(optimizer, lr, steps_per_epoch, epochs, **kwargs)


lr_schedulers = {
    'constant': get_constant_scheduler,
    'cos_ann': get_cos_ann_scheduler,
    'one_cycle': get_one_cycle_scheduler,
    'on_plateau': get_on_plateau_scheduler,
    'ann_warm_restarts': get_ann_warm_restarts_scheduler,
    'cos_ann_into_cyc_lr': get_cos_ann_into_cyc_scheduler,
}
