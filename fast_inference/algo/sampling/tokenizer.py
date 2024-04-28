import difflib
import sys


class TrajectoryTokenizer:
    nethack_obs_start_token_id = 30001
    nethack_obs_end_token_id = 30002
    nethack_obs_start_diff_token_id = 30004

    def __init__(self, tokenizer, nethack_anchor_every=1, max_ctx_tokens=8000):
        self.nethack_anchor_every = nethack_anchor_every
        self._max_ctx_tokens = max_ctx_tokens
        self.tokenizer = tokenizer

        self._observations = None
        self._actions = None
        self._token_buffer = None
        self._anchor_obs = None

        self.reset()

    def append_observation(self, observation):
        obs = observation.strip()
        i = len(self._observations)
        self._observations.append(obs)
        assert len(self._observations) == len(self._actions) + 1

        obs = obs.strip()
        if i % self.nethack_anchor_every == 0:
            # Anchor the observation (encode full observation)
            self._anchor_obs = obs
            last_obs = obs
            tokens_obs = self.tokenizer.encode(obs, add_special_tokens=False)
            tokens_obs = [self.nethack_obs_start_token_id] + tokens_obs
        else:
            diff_obs = self.diff_strings(self._anchor_obs, obs)  # todo: implement diff_strings
            last_obs = diff_obs
            tokens_obs = self.tokenizer.encode(diff_obs)
            tokens_obs = [self.nethack_obs_start_diff_token_id] + tokens_obs

        print("\033[92m <>" + last_obs + "</>\033[0m", file=sys.stderr)

        tokens_obs += [self.nethack_obs_end_token_id]

        self._token_buffer.extend(tokens_obs)

    @staticmethod
    def diff_strings(s1, s2, diffs_delimiter="\n"):
        s1 = s1.splitlines(keepends=False)
        s2 = s2.splitlines(keepends=False)
        differences = difflib.unified_diff(s1, s2, n=0)
        diff_str = ""
        for d in differences:
            if d == "--- \n" or d == "+++ \n" or d[0] == "@":
                continue
            diff_str += d + diffs_delimiter
        return diff_str

    def append_action(self, action):
        # print in red
        action = action.strip()
        print("\033[91m <>" + action + "</>\033[0m", file=sys.stderr)
        action = action.strip()
        self._actions.append(action)
        assert len(self._observations) == len(self._actions)
        tokens_action = self.tokenizer.encode(action, add_special_tokens=False)

        self._token_buffer.extend(tokens_action)

    def reset(self):
        self._observations = []
        self._actions = []
        self._token_buffer = [self.tokenizer.bos_token_id]
        self._anchor_obs = None

    def return_tokenized(self):
        assert self._token_buffer[-1] == self.nethack_obs_end_token_id
        return self._token_buffer[-self._max_ctx_tokens :]
