import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from velocity_model_based_MEConvLSTM_model import Seq2SeqMEConvLSTM
from moving_mnist_dataset import MovingMNISTDataset
from time_dependent_moving_mnist_dataset import TDMovingMNISTDataset
from channel_based_FEConvLSTM_model import Seq2SeqFEConvLSTM
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
from train_eval_utils import train_epoch, eval_epoch, eval_len_generalization, eval_velocity_generalization, ValCurveRecorder
from velocity_metrics import VelocityMetrics

class MSEPlusL1Loss(nn.Module):
    def __init__(self, mse_weight=1.0, l1_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, input_seq, target_seq):
        return self.mse_weight * self.mse(input_seq, target_seq) + self.l1_weight * self.l1(input_seq, target_seq)


def log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args):
    steps = np.arange(1, gen_pred_frames + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, gen_mean, label="MSE")
    ax.fill_between(
        steps,
        (gen_mean - gen_std).clip(min=0),
        gen_mean + gen_std,
        alpha=0.3,
        label="± std",
    )
    ax.set_xlabel("Prediction horizon t")
    ax.set_ylabel("MSE")
    ax.set_title(f"Length generalization (seq_len = {args.gen_seq_len})")
    ax.legend()
    wandb.log({"len_gen_curve_image": wandb.Image(fig)})
    plt.close(fig)

    table_data = [
        [int(step), float(m), max(0.0, float(m - s)), float(m + s)]
        for step, m, s in zip(steps, gen_mean, gen_std)
    ]
    table = wandb.Table(
        data=table_data,
        columns=["prediction_horizon", "mse", "mse_min", "mse_max"],
    )
    wandb.log({"len_gen_curve_table": table})
    wandb.log({
        "len_gen_curve_plot": wandb.plot.line(
            table,
            x="prediction_horizon",
            y="mse",
            title="Length generalization",
        )
    })


def save_len_gen_results(path, gen_mean, gen_std, details, args, epoch):
    """
    Persist one run's length-generalization results as .npz so
    plot_len_gen_comparison.py can combine several runs (models) into one
    figure with paired per-sequence statistics and qualitative strips.
    """
    run = wandb.run
    payload = dict(
        mean=gen_mean,
        std=gen_std,
        model=args.model,
        hidden_size=args.hidden_size,
        gen_input_frames=args.gen_input_frames,
        gen_seq_len=args.gen_seq_len,
        train_input_frames=args.input_frames,
        train_pred_frames=args.seq_len - args.input_frames,
        epoch=epoch,
        run_id=str(run.id) if run is not None else "none",
    )
    # per_seq_err, per_seq_err_copy_last, strip_*, and (with motion data)
    # boundary_age all come straight from eval_len_generalization.
    payload.update(details)
    np.savez_compressed(path, **payload)
    print(f"Saved length-generalization results to {path}")


def save_vel_gen_results(path, vx, vy, err, args):
    """Persist the velocity-generalization error grid for offline comparison
    (plot_vel_gen_comparison.py)."""
    run = wandb.run
    np.savez_compressed(
        path,
        vx=vx, vy=vy, err=err,
        model=args.model,
        hidden_size=args.hidden_size,
        data_v_range=args.data_v_range,
        run_id=str(run.id) if run is not None else "none",
    )
    print(f"Saved velocity-generalization results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate RNN models on Moving MNIST")
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--input_frames', type=int, default=10)
    parser.add_argument('--gen_input_frames', type=int, default=15, help='Encoder input frame count used **only** for the length-generalization experiment (separate from --input_frames, which is used for train/val/test)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes (0 = load on the main process, no parallelism)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_epochs', type=int, default=50, help='Minimum number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Enable ReduceLROnPlateau: cuts LR when val_loss stops improving')
    parser.add_argument('--lr_patience', type=int, default=5, help='Epochs with no val_loss improvement before cutting LR (only with --use_lr_scheduler)')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Multiplicative LR cut factor (only with --use_lr_scheduler)')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Floor below which LR will not be reduced further (only with --use_lr_scheduler)')
    parser.add_argument('--model', choices=['lstm', 'felstm', 'melstm'], default='felstm')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--decoder_conv_layers', type=int, default=4)
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (default: use all)')
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--v_range', type=int, default=2)
    parser.add_argument('--num_vel_modes', type=int, default=2, help='Number of velocity modes for MEConvLSTM')
    parser.add_argument('--data_v_range', type=int, default=2)
    parser.add_argument('--gen_seq_len', type=int, default=40, help='Sequence length used **only** for length‑generalization evaluation (must be > seq_len)')
    parser.add_argument('--data_seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--model_seed', type=int, default=None, help='Random seed for model initialization (default: random)')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the run')
    parser.add_argument('--model_save_dir', type=str, default='./fernn/movmnist/', help='Directory to save model checkpoints')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model checkpoint to load for evaluation')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate the loaded model without training')
    parser.add_argument('--gen_vel_min', type=int, default=-2, help='Minimum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_max', type=int, default= 2, help='Maximum integer pixel velocity evaluated along each axis')
    parser.add_argument('--gen_vel_step', type=int, default= 1, help='Step size (integer) for the velocity grid')
    parser.add_argument('--gen_vel_n_seq', type=int, default=128, help='How many sequences to sample **per (vx,vy)** for velocitygeneralisation test')
    parser.add_argument('--run_velocity_generalization', action='store_true', help='Run velocity generalization test (default: False)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0, None for no clipping)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default="FERNN", help='Wandb project')
    parser.add_argument('--wandb_dir', type=str, default='./tmp/', help='Wandb directory')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb name')
    parser.add_argument("--check_velocity_predictor",action="store_true",help="Debug the velocity predictor during training/evaluation.")
    parser.add_argument('--val_curve_interval', type=int, default=25, help='Record validation loss every N training batches for the loss-vs-steps curve (0 = off)')
    parser.add_argument('--val_curve_size', type=int, default=256, help='Number of fixed validation sequences used for the loss-vs-steps curve')
    parser.add_argument('--eval_velocity_mode', choices=['frozen', 'tracked', 'both'], default='frozen',
                        help="MELSTM decoder velocity at evaluation: 'frozen' = honest inference (default, drives scheduler/selection); "
                             "'tracked' = oracle GT-tracked eval drives scheduler/selection; "
                             "'both' = select on frozen but also log val_tracked_loss each epoch")
    parser.add_argument('--len_gen_every', type=int, default=0,
                        help='Also run length-generalization every N epochs regardless of val improvement '
                             '(0 = only on new best val). Saves to a separate *_latest.npz')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint_*.pth written during a previous run: restores model, '
                             'optimizer, scheduler, epoch, best-val, loss curves, and continues the same '
                             'wandb run. (Unlike --load_model, which loads best weights only.)')
    args = parser.parse_args()

    # --check_velocity_predictor only makes the datasets return motion
    # (return_motion=...): the shared train/eval functions detect it in the
    # batch and add velocity metrics / state visualizations on top of the
    # identical training procedure.
    train_fn = train_epoch
    eval_fn = eval_epoch
    len_gen_fn = eval_len_generalization


    # Set seeds for reproducibility
    # Data seed for consistent dataset splits
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    random.seed(args.data_seed)

    # Create model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Load the resume checkpoint before wandb.init so the original run
    # (and its id-based file names) can be continued.
    resume_ckpt = None
    if args.resume is not None:
        print(f"Loading resume checkpoint from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location='cpu')

    # Initialize wandb
    wandb.init(entity=args.wandb_entity,
               project=args.wandb_project,
               dir=args.wandb_dir,
               config=vars(args),
               name=args.wandb_name,
               id=resume_ckpt["run_id"] if resume_ckpt else None,
               resume="allow" if resume_ckpt else None)

    assert args.input_frames < args.seq_len, "input_frames must be less than seq_len"
    assert args.gen_input_frames < args.gen_seq_len, "gen_input_frames must be less than gen_seq_len"
    pred_frames = args.seq_len - args.input_frames
    gen_pred_frames = args.gen_seq_len - args.gen_input_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    

    # # Dataset and splits
    # train_dataset = MovingMNISTDataset(
    #     root=args.root,
    #     train=True,
    #     seq_len=args.seq_len,
    #     image_size=args.image_size,
    #     velocity_range_x=(-args.data_v_range,args.data_v_range),
    #     velocity_range_y=(-args.data_v_range,args.data_v_range),
    #     num_digits=2
    # )
    
    # test_dataset = MovingMNISTDataset(
    #     root=args.root,
    #     train=False,
    #     seq_len=args.seq_len,
    #     image_size=args.image_size,
    #     velocity_range_x=(-args.data_v_range,args.data_v_range),
    #     velocity_range_y=(-args.data_v_range,args.data_v_range),
    #     num_digits=2
    # )

    # gen_test_dataset = MovingMNISTDataset(
    #         root=args.root,
    #         train=False,
    #         seq_len=args.gen_seq_len,
    #         image_size=args.image_size,
    #         velocity_range_x=(-args.data_v_range, args.data_v_range),
    #         velocity_range_y=(-args.data_v_range, args.data_v_range),
    #         num_digits=2,
    #         random=False
    # )

    train_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=True,
        seq_len=args.seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode="piecewise",
        transition_mode="smooth",

        min_segment=3,
        max_segment=6,

        p_change=0.25,
        smooth_probability=0.8,

        motion_difficulty=None,
        freeze_after=args.input_frames,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor,
        return_positions=False,

        transform=None,
        download=True,

        random=True,
        seed=42,

        max_tries=300,
    )

    test_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode="piecewise",
        transition_mode="smooth",

        min_segment=3,
        max_segment=6,

        p_change=0.25,
        smooth_probability=0.8,

        motion_difficulty=None,
        freeze_after=args.input_frames,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor,
        return_positions=False,

        transform=None,
        download=True,

        random=False,  # Fixed benchmark set (same idea as gen_test_dataset)
        seed=123,      # distinct seed so test and len-gen draw different streams

        max_tries=300,
    )

    gen_test_dataset = TDMovingMNISTDataset(
        root=args.root,
        train=False,
        seq_len=args.gen_seq_len,
        num_digits=2,
        image_size=args.image_size,
        max_speed=args.data_v_range,

        motion_mode="piecewise",
        transition_mode="smooth",

        min_segment=3,
        max_segment=6,

        p_change=0.25,
        smooth_probability=0.8,

        motion_difficulty=None,

        freeze_after=args.gen_input_frames,

        min_center_distance=20,
        reject_overlap=True,
        require_distinct_velocities=True,

        return_motion=args.check_velocity_predictor,
        return_positions=False,

        transform=None,
        download=True,

        random=False, 
        seed=42,
        max_tries=300
    )

    # Split training data into train and validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(args.data_seed))
    
    # Limit training samples if specified
    if args.max_train_samples is not None and args.max_train_samples < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.data_seed))[:args.max_train_samples]
        train_ds = Subset(train_ds, indices)
        print(f"Limited training to {args.max_train_samples} samples")
    
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kwargs)
    # persistent_workers=False on purpose for the two fixed benchmark sets:
    # their seeded RNGs are stateful, and persistent workers would carry
    # advanced RNG state into the next evaluation (different sequences per
    # call). Non-persistent workers re-fork from the parent dataset — whose
    # RNG is reset via reset_rng() before each evaluation — so every pass
    # sees the identical set.
    fixed_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, **fixed_loader_kwargs)
    gen_test_loader = DataLoader(gen_test_dataset,
                                batch_size=args.batch_size, **fixed_loader_kwargs)


    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_dataset)}")
    print(f"Length‑gen test size: {len(gen_test_dataset)} | Eval sequence length: {args.gen_seq_len}")
    print(f"Input frames: {args.input_frames}, Pred frames: {pred_frames}")
    print(f"Gen input frames: {args.gen_input_frames}, Gen pred frames: {gen_pred_frames} (gen_seq_len={args.gen_seq_len})")
    print(f"Model seed: {args.model_seed}")
    print(f"Data seed: {args.data_seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"LR scheduler: {'ReduceLROnPlateau' if args.use_lr_scheduler else 'none'}")
    if args.use_lr_scheduler:
        print(f"  patience={args.lr_patience}  factor={args.lr_factor}  min_lr={args.lr_min}")
    print(f"Model: {args.model}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Decoder conv layers: {args.decoder_conv_layers}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Image size: {args.image_size}")
    print(f"Velocity range: {args.v_range}")
    v_list = [(x, y) for x in range(-args.v_range, args.v_range + 1) for y in range(-args.v_range, args.v_range + 1)]
    num_v = len(v_list)
    print(f"Num v: {num_v}")

    # Set model initialization seed
    if args.model_seed is None:
        args.model_seed = int(time.time()) % 10000  # Use current time as seed if not provided
        wandb.config.update({"model_seed": args.model_seed}, allow_val_change=True)
    
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    
    if args.model == "felstm":
        model = Seq2SeqFEConvLSTM(
                input_channels=1,
                hidden_channels=args.hidden_size,
                kernel_size=args.kernel_size,
                v_range=args.v_range,
                pool_type='max',
                decoder_conv_layers=args.decoder_conv_layers
            ).to(device)
    elif args.model == "lstm":
        assert args.v_range == 0, "v_range must be 0 for grnn"
        model = Seq2SeqFEConvLSTM(
                input_channels=1,
                hidden_channels=args.hidden_size,
                kernel_size=args.kernel_size,
                v_range=0,
                pool_type='max',
                decoder_conv_layers=args.decoder_conv_layers
            ).to(device)
    elif args.model == "melstm":
        model = Seq2SeqMEConvLSTM(
                input_channels=1,
                hidden_channels=args.hidden_size,
                kernel_size=args.kernel_size,
                n_slots= args.num_vel_modes,
                slot_reduce = 'max',
                decoder_layers = args.decoder_conv_layers
            ).to(device)

    # Parameter count: printed per top-level submodule and stored in the
    # wandb config so runs of different models are directly comparable.
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters ({args.model}): {num_params:,} total, {num_trainable:,} trainable")
    for name, module in model.named_children():
        sub = sum(p.numel() for p in module.parameters())
        if sub > 0:
            print(f"  {name:<20s} {sub:>12,}")
    wandb.config.update(
        {"num_parameters": num_params, "num_trainable_parameters": num_trainable},
        allow_val_change=True,
    )

    # Load model if specified
    if args.load_model is not None:
        print(f"Loading model from {args.load_model}")
        try:
            model.load_state_dict(torch.load(args.load_model))
            print(f"Successfully loaded model weights.")
        except Exception as e:
            print(f"Failed to load model weights {e}")

    # If evaluate_only is True, skip training and only run evaluation
    if args.evaluate_only:
        print("Running evaluation only...")
        criterion = MSEPlusL1Loss()
        test_dataset.reset_rng()       # identical benchmark set every evaluation
        test_loss = eval_fn(model, test_loader, criterion, device, args.input_frames, 0, split_name="test")
        print(f"Test Loss: {test_loss:.4f}")

        wandb.log({
            f"test_loss": test_loss,
            "epoch": 0
        })

        print("Running length generalization test...")
        gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
        gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames, subsample_t=2)
        print(f"Length generalization mean MSE: {gen_mean.mean():.4f}")
        save_len_gen_results(
            os.path.join(args.model_save_dir, f"len_gen_{args.model}_{wandb.run.id}.npz"),
            gen_mean, gen_std, gen_details, args, epoch=0,
        )

        wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
        wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
        wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})
        log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args)

        if args.run_velocity_generalization:
            print("Running velocity generalization test...")
            vx, vy, err = eval_velocity_generalization(model, device, args)
            save_vel_gen_results(
                os.path.join(args.model_save_dir, f"vel_gen_{args.model}_{wandb.run.id}.npz"),
                vx, vy, err, args,
            )
            print("Velocity generalization results:")
            for i, vx_i in enumerate(vx):
                for j, vy_j in enumerate(vy):
                    print(f"Velocity (vx={vx_i}, vy={vy_j}): MSE {err[j, i]:.4f}")
            
            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                    for i, vx_i in enumerate(vx)
                    for j, vy_j in enumerate(vy)})

            fig, ax = plt.subplots()
            im = ax.imshow(err[::-1],  # flip y so +vy is up
                        extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                        origin='lower')
            ax.set_xlabel('$v_x$  (pixels / frame)')
            ax.set_ylabel('$v_y$')
            ax.set_title('Velocity-generalisation MSE')
            fig.colorbar(im, ax=ax)
            wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
            plt.close(fig)

        return

    # Optimizers & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MSEPlusL1Loss()

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min,
        )

    # Fine-grained loss-vs-training-batches curves (fixed val subset,
    # materialized once — see ValCurveRecorder)
    curve_recorder = None
    if args.val_curve_interval > 0:
        print(f"Materializing {args.val_curve_size} fixed val sequences for the loss curve...")
        curve_recorder = ValCurveRecorder(
            val_ds, args.val_curve_size, args.val_curve_interval,
            args.input_frames, device,
        )

    history_path = os.path.join(args.model_save_dir, f"history_{args.model}_{wandb.run.id}.json")

    def dump_history(history):
        payload = {"config": vars(args), "history": history}
        if curve_recorder is not None:
            payload.update(curve_recorder.as_dict())
        with open(history_path, "w") as f:
            json.dump(payload, f, indent=2)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'test_loss': [], 'epoch_time_sec': []}
    best_val_losses = float('inf')
    start_epoch = 1
    checkpoint_path = os.path.join(args.model_save_dir, f"checkpoint_{args.model}_{wandb.run.id}.pth")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if scheduler is not None and resume_ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        best_val_losses = resume_ckpt["best_val_losses"]
        history = resume_ckpt["history"]
        if curve_recorder is not None and resume_ckpt.get("curve_recorder") is not None:
            curve_recorder.load_state_dict(resume_ckpt["curve_recorder"])
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from epoch {resume_ckpt['epoch']} "
              f"(continuing at {start_epoch}) | best val so far: {best_val_losses:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_fn(model, train_loader, optimizer, criterion, device, args.input_frames, args.grad_clip,
                              curve_recorder=curve_recorder)
        epoch_time = time.time() - epoch_start
        val_loss = eval_fn(model, val_loader, criterion, device, args.input_frames, epoch, split_name="val")

        # Oracle (GT-tracked decoder velocity) validation: measures the model
        # without velocity-estimation drift; the gap to val_loss isolates
        # velocity error from rendering error. MELSTM-only effect.
        val_tracked_loss = None
        if args.eval_velocity_mode in ('tracked', 'both'):
            val_tracked_loss = eval_fn(model, val_loader, criterion, device, args.input_frames, epoch,
                                       split_name="val_tracked", decoder_velocity_mode="tracked")

        # Metric driving the scheduler / best-model selection / len-gen gating
        selection_val = val_tracked_loss if args.eval_velocity_mode == 'tracked' else val_loss

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time_sec'].append(epoch_time)
        tracked_str = f" | Val (tracked): {val_tracked_loss:.4f}" if val_tracked_loss is not None else ""
        print(f"Epoch {epoch}/{args.epochs} | {args.model:^8} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{tracked_str} | {epoch_time:.1f}s")

        if scheduler is not None:
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(selection_val)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"  LR reduced: {lr_before:.2e} -> {lr_after:.2e}")

        # Log metrics to wandb
        log_payload = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time_sec": epoch_time,
            "epoch": epoch
        }
        if val_tracked_loss is not None:
            log_payload["val_tracked_loss"] = val_tracked_loss
        wandb.log(log_payload)

        # Persist curves incrementally so a killed run still leaves data
        dump_history(history)
        
        # Save model if it's the best so far
        ran_len_gen_this_epoch = False
        if selection_val < best_val_losses:
            best_val_losses = selection_val
            ran_len_gen_this_epoch = True
            model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
            model_path = os.path.join(args.model_save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model for {args.model} at {model_path}")
            
            # Evaluate on test set with best model
            test_dataset.reset_rng()   # identical benchmark set every evaluation
            test_loss = eval_fn(model, test_loader, criterion, device, args.input_frames, epoch, split_name="test")

            history['test_loss'].append(test_loss)
            print(f"Test Loss: {test_loss:.4f}")

            gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
            gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames)
            save_len_gen_results(
                os.path.join(args.model_save_dir, f"len_gen_{args.model}_{wandb.run.id}.npz"),
                gen_mean, gen_std, gen_details, args, epoch=epoch,
            )

            wandb.log({f"len_gen_mean_t{t+1}": gen_mean[t] for t in range(len(gen_mean))})
            wandb.log({f"len_gen_std_t{t+1}":  gen_std[t]  for t in range(len(gen_std))})
            wandb.log({f"len_gen_mean_mean_over_time": gen_mean.mean()})
            log_length_generalization_curve(gen_mean, gen_std, gen_pred_frames, args)
            
            if args.run_velocity_generalization:
                vx, vy, err = eval_velocity_generalization(model, device, args)
                save_vel_gen_results(
                    os.path.join(args.model_save_dir, f"vel_gen_{args.model}_{wandb.run.id}.npz"),
                    vx, vy, err, args,
                )

                wandb.log({f"vel_gen_err_vx{vx_i}_vy{vy_j}": err[j, i]
                        for i, vx_i in enumerate(vx)
                        for j, vy_j in enumerate(vy)})

                fig, ax = plt.subplots()
                im = ax.imshow(err[::-1],  # flip y so +vy is up
                            extent=[vx[0]-0.5, vx[-1]+0.5, vy[0]-0.5, vy[-1]+0.5],
                            origin='lower')
                ax.set_xlabel('$v_x$  (pixels / frame)')
                ax.set_ylabel('$v_y$')
                ax.set_title('Velocity-generalisation MSE')
                fig.colorbar(im, ax=ax)
                wandb.log({"vel_gen_heatmap": wandb.Image(fig)})
                plt.close(fig)

            # Log model path and metrics to wandb
            wandb.log({
                "best_model_path": model_path,
                "best_val_loss": val_loss,
                "test_loss": test_loss,
                "epoch": epoch
            })

        # Periodic length generalization, decoupled from val improvement:
        # with honest (frozen-velocity) validation the selection metric can
        # saturate while the model keeps improving, which would otherwise
        # starve the len-gen monitoring. Saves to *_latest.npz so it never
        # overwrites the best-checkpoint results.
        if (args.len_gen_every > 0 and epoch % args.len_gen_every == 0
                and not ran_len_gen_this_epoch):
            gen_test_dataset.reset_rng()   # identical benchmark set every evaluation
            gen_mean, gen_std, gen_details = len_gen_fn(model, gen_test_loader, device, args.gen_input_frames)
            save_len_gen_results(
                os.path.join(args.model_save_dir, f"len_gen_{args.model}_{wandb.run.id}_latest.npz"),
                gen_mean, gen_std, gen_details, args, epoch=epoch,
            )
            wandb.log({"len_gen_mean_over_time_latest": gen_mean.mean(), "epoch": epoch})

        # Full resume checkpoint (model + optimizer + scheduler + curves),
        # written atomically so a crash mid-save can't corrupt the file.
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val_losses": best_val_losses,
            "history": history,
            "curve_recorder": curve_recorder.state_dict() if curve_recorder is not None else None,
            "run_id": wandb.run.id,
            "args": vars(args),
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, checkpoint_path)

    # Final history dump (also written incrementally after every epoch)
    dump_history(history)
    print(f"Saved training history to {history_path}")

    # Print final best model paths and test results
    print("\nBest model paths and test results:")
    model_filename = f"{args.model}_best_model_{wandb.run.id}.pth"
    model_path = os.path.join(args.model_save_dir, model_filename)
    test_loss = history['test_loss'][-1] if history['test_loss'] else None
    print(f"{args.model}: {model_path} | Test Loss: {test_loss:.4f}" if test_loss is not None else f"{args.model}: {model_path} | Test Loss: N/A")

if __name__ == '__main__':
    main()
