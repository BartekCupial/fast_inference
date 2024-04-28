from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from fast_inference.cfg.arguments import parse_full_cfg, parse_sf_args
from fast_inference.envs.env_utils import register_env
from fast_inference.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from fast_inference.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from fast_inference.algo.sampling.tokenizer import TrajectoryTokenizer
from fi_examples.nethack.nethack_env import make_nethack_env, NETHACK_ENVS
from fi_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_tokenizer,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)

def parse_nethack_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_nethack_env(parser)
    add_extra_params_tokenizer(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def register_nethack_envs():
    for env_name in NETHACK_ENVS.keys():
        register_env(env_name, make_nethack_env)


def register_nethack_components():
    register_nethack_envs()


def main():
    register_nethack_components()
    cfg = parse_nethack_args()
    
    vector_size = cfg.num_envs_per_worker
    num_splits = cfg.worker_num_splits
    num_envs = vector_size // num_splits
    
    env_info = obtain_env_info_in_a_separate_process(cfg)
    
    runner = NonBatchedVectorEnvRunner(
        cfg, 
        env_info,
        num_envs=num_envs,
        worker_idx=0,
        split_idx=0,
    )
    
    policy_inputs = runner.generate_policy_inputs()
    
    sampling_params = SamplingParams(
        max_tokens=4096,
    )
    llm = LLM(model="facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    trajectory_tokenizer = TrajectoryTokenizer(
        tokenizer, nethack_anchor_every=cfg.nethack_anchor_every, max_ctx_tokens=cfg.max_ctx_tokens
    )
    llm = LLM(model=cfg.model)
    prompt_token_ids = trajectory_tokenizer.encode()
    outputs = llm.generate(sampling_params, prompt_token_ids)
    
    policy_outputs = query_model(policy_inputs)
    runner.process_policy_outputs(policy_outputs)
    episodic_stats = runner.advance_rollouts()
    
    
    
if __name__ == "__main__":
    main()