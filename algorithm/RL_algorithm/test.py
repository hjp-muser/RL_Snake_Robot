import sys
from importlib import import_module
import os.path as osp
import numpy as np

from baselines import logger
from baselines.common.vec_env import VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from algorithm.RL_algorithm.utils.tensorflow1.env_utils import get_env_type, build_env
from algorithm.RL_algorithm.utils.arg_utils import common_arg_parser, parse_unknown_args

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def parse_cmdline_kwargs(args):
    """
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    """

    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def get_alg_module(alg, submodule='model'):
    try:
        # first try to import the alg module from local directory
        print('.'.join(['algorithm', 'RL_algorithm', alg,  submodule]))
        alg_module = import_module('.'.join(['algorithm', 'RL_algorithm', alg, submodule]))
    except ImportError:
        # import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module


def get_algorithm_model(alg):
    return get_alg_module(alg).Model


def get_model_defaults(alg, env_id):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_id)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def train(args, extra_args):  # TODO: pretrain
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    # load model
    alg_model = get_algorithm_model(args.alg)
    # load default super-parameter
    alg_kwargs = get_model_defaults(args.alg, env_id)
    alg_kwargs.update(extra_args)

    # build env
    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    # choose network
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    # model instance
    model_ins = alg_model(
        env=env,
        seed=seed,
        **alg_kwargs
    )
    model_ins.learn(total_timesteps)

    return model_ins, env


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model


if __name__ == '__main__':
    sys_args = ['--env=reach_target-state-v0', '--num_env=2', '--alg=a2c', '--network=mlp', '--num_timesteps=2.5e4',
                '--gamma=0.5', '--max_grad_norm=2']
    sys_args.extend(sys.argv)
    main(sys_args)
