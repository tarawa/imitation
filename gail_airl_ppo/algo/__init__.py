from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .wgail import WGAIL

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'wgail': WGAIL
}
