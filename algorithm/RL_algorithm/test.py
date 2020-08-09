# import os
# import sys
# dirpath = os.path.dirname(os.path.abspath(__file__))
# for _ in range(4):
#     print(dirpath)
#     sys.path.append(dirpath)
#     dirpath = os.path.split(dirpath)[0]

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

    total_timesteps = int(args.total_timesteps)

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
        # model.load_newest()
        model.load_index(0)
        # for i in range(len(env.envs)):
        #     env.envs[i].load_env_param()
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, state=state, mask=dones)
            else:
                actions, _, _, _ = model.step(obs)
            # print("actions = ", actions)
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
    # A2C
    a2c_args = ['--env=reach_target-state-param-v0', '--num_env=2', '--alg=a2c', '--network=mlp', '--ent_coef=0.1',
                '--total_timesteps=1e7', '--seed=10', '--gamma=0.9', '--max_grad_norm=2', "--tb_log_path='./a2c'"]

    a2c_play = ['--env=reach_target-state-param-v0', '--alg=a2c', '--network=mlp', '--total_timesteps=0',
                '--seed=10', '--play']
    ###########################################################################################################
    # SAC
    sac_args = ['--env=reach_target-state-param-v0', '--num_env=1', '--alg=sac', '--network=mlp', '--total_timesteps=6e6',
                '--gamma=0.9', '--buffer_size=50000', '--learning_start_threshold=200', "--ent_coef='auto'",
                '--batch_size=128', '--tau=0.05'] # , "--tensorboard_log_path='./sac'"

    sac_play = ['--env=reach_target-state-param-v0', '--alg=sac', '--network=mlp', '--total_timesteps=0',
                '--seed=10', '--play']

    ###########################################################################################################
    multi_a2c_args = ['--env=reach_target-state-param-v1', '--num_env=2', '--alg=multi_a2c', '--network=mlp',
                      '--total_timesteps=1e6', '--seed=10', '--gamma=0.9', '--max_grad_norm=0.2',
                      '--ent_coef=0.0005', "--tb_log_path='./multi_a2c'"]

    multi_a2c_play = ['--env=reach_target-state-param-v1', '--alg=multi_a2c', '--network=mlp', '--total_timesteps=0',
                      '--seed=10', '--play']

    ###########################################################################################################
    hac_args = ['--env=reach_target-state-param-v0', '--alg=hac', '--total_timesteps=1e6', '--seed=10']
    hac_play = []

    ###########################################################################################################
    ddpg_args = ['--env=reach_target-state-param-v0', '--alg=ddpg', '--total_timesteps=1e6']
    ddpg_play = ['--env=reach_target-state-param-v0', '--alg=ddpg', '--total_timesteps=0', '--play']

    ##########################################################################################################
    ###########################################################################################################
    ppo_args = ['--env=reach_target-state-param-v0', '--alg=ppo', '--total_timesteps=1e6']
    ppo_play = ['--env=reach_target-state-param-v0', '--alg=ppo', '--total_timesteps=0', '--play']

    main(ppo_play)
