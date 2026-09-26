"""
Dataset registry for train_motion.py.

Each entry is a class with two static methods:

    add_args(parser)  -- its own command-line options (prefixed, so they cannot collide)
    build(args)       -- dict(train=..., val=..., test=..., gen_test=... or None,
                              extra_tests={name: dataset}, meta=dict)

Shared data options (input/pred frames, split sizes, seeds, image size) live in train_motion.py
and are read here from `args`. Splits follow the Moving MNIST protocol: training data are drawn
fresh every epoch (random=True); val/test/gen_test are fixed benchmarks; the rollout is driven
by freeze_after = input_frames (the velocity freezes at the last transition in the context) unless
--no_freeze_future is given.
"""
REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def _floats(s):
    if s is None or s == "":
        return None
    vals = [float(v) for v in str(s).split(",")]
    return tuple(vals) if len(vals) > 1 else vals[0]


def _ints(s):
    if s is None or s == "":
        return None
    return tuple(int(v) for v in str(s).split(","))


def split_seeds(args):
    base = int(args.data_seed)
    return dict(train=base, val=base + 1, test=base + 2, gen=base + 3, extra=base + 4)


def freeze(args, input_frames=None):
    if getattr(args, "no_freeze_future", False):
        return None
    return int(input_frames if input_frames is not None else args.input_frames)
