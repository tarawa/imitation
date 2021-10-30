from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .wgail import WGAIL
from .bc import BC

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'wgail': WGAIL,
    'bc': BC
}
