# TODO: Player-Level Futsal MARL and Football Visualization

## Goal

Build enough working knowledge to train a futsal-like, player-level football policy:

- five simultaneous learning agents per team;
- one logical agent for every player, including the goalkeeper;
- simultaneous actions through PettingZoo's `ParallelEnv` API;
- parameter-shared policies trained through self-play;
- headless training with recorded evaluation matches; and
- a persistent rating ladder for comparing policy checkpoints.

GRF's stock `5_vs_5` scenario is only futsal-like. It uses the football engine's field geometry and rules, and its goalkeepers are not controllable. The target therefore requires a custom 5v5 scenario with ten controllable players. Treat faithful futsal dimensions and rules as a separate simulator-validation question.

## Working decisions

- [ ] Learn and use the Gymnasium single-agent API before starting PettingZoo.
- [ ] Use the PettingZoo parallel API, since both teams act on the same game tick.
- [ ] Represent ten logical agents: `left_0` through `left_4` and `right_0` through `right_4`.
- [ ] Start with parameter-shared IPPO to validate the pipeline, then use MAPPO with a centralized critic for the futsal target.
- [ ] Share actor weights initially, with team, role, and controlled-player identity included in the observation.
- [ ] Decide from evidence whether goalkeepers need a separate shared goalkeeper policy.
- [ ] Keep the modern MARL environment separate from GRF's older Python and Gym dependencies.
- [ ] Train without rendering and render only evaluation episodes.
- [ ] Use GRF raw observations and actions `0` through `18`.

## 1. Learn the Gymnasium API

- [ ] Create an isolated modern RL environment and install `gymnasium[classic-control]`.
- [ ] Read the [Gymnasium basic usage guide](https://gymnasium.farama.org/introduction/basic_usage/).
- [ ] Run [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) with random actions.
- [ ] Inspect and explain:
  - `env.observation_space`;
  - `env.action_space`;
  - the `(observation, info)` result from `reset()`; and
  - the `(observation, reward, terminated, truncated, info)` result from `step()`.
- [ ] Handle `terminated` and `truncated` separately, and reset when either is true.
- [ ] Seed the environment and action space, then reproduce the same random rollout twice.
- [ ] Render CartPole with both `human` and `rgb_array` modes.
- [ ] Add `RecordEpisodeStatistics` and record episode return and length.
- [ ] Add `RecordVideo` and save a complete evaluation episode.
- [ ] Write one observation wrapper and one reward wrapper without modifying CartPole itself.
- [ ] Run several CartPole instances with `gymnasium.make_vec()` or `SyncVectorEnv` and inspect the batched shapes.
- [ ] Train a small PPO policy and compare it with the random baseline on fixed evaluation seeds.
- [ ] Follow the [custom environment tutorial](https://gymnasium.farama.org/main/tutorials/environment_creation/) and implement a small single-agent environment.
- [ ] Make the custom environment pass Gymnasium's environment checker.

Completion check: the custom environment is seeded, wrapped, vectorized, rendered, and checked successfully, and the trained CartPole policy beats random actions after reload.

## 2. Understand the PettingZoo parallel API

- [ ] Add PettingZoo, MPE2, PyTorch, and TensorBoard to the modern RL environment used for Gymnasium.
- [ ] Run a random-policy episode in [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/).
- [ ] Write down how PettingZoo changes the Gymnasium contract from one observation and action into dictionaries keyed by agent ID.
- [ ] Print and inspect these values after `reset()` and `step()`:
  - `env.agents`
  - observations keyed by agent ID
  - actions keyed by agent ID
  - per-agent rewards
  - terminations and truncations
- [ ] Render a complete Simple Spread episode with `render_mode="human"`.
- [ ] Confirm that every live agent supplies exactly one action per call to `step()`.
- [ ] Write down the tensor layout used by the rollout buffer. Prefer `time x environment x agent x feature`.

Completion check: five random episodes run without API errors, and the observation/action shapes are understood.

## 3. Train a small cooperative policy

- [ ] Adapt the [PettingZoo CleanRL PPO tutorial](https://pettingzoo.farama.org/main/tutorials/cleanrl/implementing_PPO/) to Simple Spread.
- [ ] Use one shared actor and value network for all agents.
- [ ] Add deterministic seeds and separate training and evaluation seeds.
- [ ] Log:
  - team return;
  - episode length;
  - policy and value losses;
  - entropy;
  - approximate KL divergence; and
  - steps per second.
- [ ] Save checkpoints and a short rendered evaluation episode.
- [ ] Compare the trained policy with a random policy on the same evaluation seeds.

Completion check: the learned policy beats the random baseline on held-out seeds and can be reloaded from a checkpoint.

## 4. Train parallel pixel agents in Space Invaders

- [ ] Install the PettingZoo Atari dependencies in the modern MARL environment and install the required ROMs through AutoROM.
- [ ] Read the [PettingZoo Space Invaders documentation](https://pettingzoo.farama.org/main/environments/atari/space_invaders/).
- [ ] Run `space_invaders_v2.parallel_env()` with random actions for both agents.
- [ ] Confirm the environment exposes two agents, six discrete actions, and RGB observations.
- [ ] Render a full episode and record an `rgb_array` rollout.
- [ ] Apply standard Atari preprocessing explicitly:
  - maximum over adjacent observations for flicker;
  - frame skipping;
  - grayscale and resize;
  - frame stacking; and
  - agent identity indicator.
- [ ] Adapt the [advanced PettingZoo CleanRL PPO tutorial](https://pettingzoo.farama.org/main/tutorials/cleanrl/advanced_PPO/) to Space Invaders.
- [ ] Compare independent actor weights with a parameter-shared actor.
- [ ] Track both individual returns and combined return because Space Invaders mixes cooperation with competition.
- [ ] Save a checkpoint and a video showing both trained agents.

Completion check: the trained agents outperform random agents on fixed evaluation seeds, and the saved model reproduces its evaluation score after reload.

## 5. Add adversarial self-play

- [ ] Move to [MPE2 Simple Tag](https://mpe2.farama.org/main/environments/simple_tag/).
- [ ] Train one side against a fixed random or scripted opponent.
- [ ] Freeze policy snapshots instead of updating both opponents inside the same rollout.
- [ ] Build a small opponent pool containing:
  - the current policy;
  - recent snapshots;
  - older snapshots; and
  - fixed baseline opponents.
- [ ] Track wins, losses, draws where applicable, and performance by opponent.
- [ ] Confirm that a new checkpoint does not regress against older snapshots before promoting it.

Completion check: the policy beats the fixed baseline and remains competitive against several frozen snapshots.

## 6. Learn GRF visualization and replay tools

- [ ] Read [gfootball/play_game.py](gfootball/play_game.py).
- [ ] Read [gfootball/doc/saving_replays.md](gfootball/doc/saving_replays.md).
- [ ] Play the small academy scenario:

  ```bash
  python3 -m gfootball.play_game \
    --level=academy_3_vs_1_with_keeper \
    --action_set=full
  ```

- [ ] Watch two built-in controllers:

  ```bash
  python3 -m gfootball.play_game \
    --level=5_vs_5 \
    --players="bot:left_players=1;bot:right_players=1"
  ```

- [ ] Stop `play_game` with Ctrl+C and locate the resulting dump under `/tmp/dumps`.
- [ ] Replay a trace:

  ```bash
  python3 -m gfootball.replay --trace_file=/tmp/dumps/TRACE.dump
  ```

- [ ] Convert a trace to video:

  ```bash
  python3 -m gfootball.dump_to_video --trace_file=/tmp/dumps/TRACE.dump
  ```

- [ ] Create a programmatic smoke test that calls `env.render(mode="rgb_array")` and verifies that it returns an RGB NumPy array.
- [ ] Record one complete evaluation episode without enabling rendering during training.

Completion check: one trace can be replayed, converted to video, and inspected frame by frame.

## 7. Create the futsal-like GRF scenario

- [ ] Read [gfootball/scenarios/5_vs_5.py](gfootball/scenarios/5_vs_5.py).
- [ ] Add a separate custom scenario rather than changing the stock `5_vs_5` scenario.
- [ ] Make all five players on each team controllable, including both goalkeepers.
- [ ] Give every player a stable role and starting position.
- [ ] Disable or modify football rules that conflict with the intended futsal task where GRF supports doing so.
- [ ] Document which futsal rules and pitch properties GRF cannot represent.
- [ ] Verify that GRF returns ten observations and accepts ten ordered actions on every step.
- [ ] Run random-action episodes with all ten players controlled.

Completion check: the custom scenario repeatedly completes episodes with ten externally controlled players and no action-count errors.

## 8. Wrap GRF as a PettingZoo parallel environment

- [ ] Expose ten agents: five per team.
- [ ] Define `possible_agents`, `agents`, `observation_space(agent)`, and `action_space(agent)`.
- [ ] Use `Discrete(19)` for every action space.
- [ ] Maintain an explicit, tested mapping from PettingZoo agent IDs to GRF's ordered observations and actions.
- [ ] Convert an action dictionary into GRF's ordered action list. Do not depend on dictionary iteration order.
- [ ] Convert GRF's observation list into observations keyed by player agent ID.
- [ ] Include role and team identity in each encoded observation.
- [ ] Provide a global `state()` suitable for MAPPO's centralized critic.
- [ ] Give each player its team's shared reward. Keep opposing team rewards zero-sum.
- [ ] Preserve GRF's mirrored observations and test the transformation for every right-team player.
- [ ] Decide whether actors receive the full raw state or a player-centered partial observation. Record this as part of the experiment contract.
- [ ] Handle reset, termination, truncation, and cleanup without leaking state between episodes.
- [ ] Run PettingZoo's `parallel_api_test` against the wrapper.
- [ ] Add regression tests for:
  - all ten observation and action mappings;
  - goalkeeper mapping;
  - reward signs and team sharing;
  - side mirroring;
  - stable agent IDs; and
  - episode reset.

Completion check: the wrapper passes `parallel_api_test`, and ten random agents can finish repeated matches without mapping or invalid-action failures.

## 9. Train the first player-level football agents

- [ ] Start with a small player-level academy scenario before full 5v5.
- [ ] Use one logical agent per controlled player from the first football experiment onward.
- [ ] Begin with a parameter-shared actor and role embeddings.
- [ ] Train IPPO first as a plumbing baseline.
- [ ] Add a centralized critic and compare MAPPO against the IPPO baseline on the same seeds and opponent pool.
- [ ] Begin with GRF checkpoint shaping, then reduce shaping as the policy improves.
- [ ] Replicate team reward to all five teammates and keep individual shaping small enough that it cannot dominate winning.
- [ ] Evaluate using match outcomes rather than shaped training return.
- [ ] Save frozen opponent snapshots throughout training.
- [ ] Render fixed-seed evaluation matches after each promoted checkpoint.
- [ ] Progress to the custom 5v5 scenario only after the academy policy clearly beats random and scripted baselines.
- [ ] Inspect whether players specialize by role despite sharing actor weights.
- [ ] Compare a shared goalkeeper policy head with a separate goalkeeper actor.

Completion check: a saved football policy reloads correctly, beats the chosen baseline, and produces a watchable evaluation video.

## 10. Build an Elo-like evaluation ladder

- [ ] Treat every promoted policy checkpoint as an immutable ladder competitor.
- [ ] Give every competitor a stable ID tied to its checkpoint digest and training metadata.
- [ ] Store every match with:
  - competitor IDs;
  - random seed;
  - assigned side;
  - final score;
  - win, draw, or loss outcome;
  - termination reason; and
  - simulator and observation-encoder versions.
- [ ] Play paired fixtures with the same seed and swapped sides.
- [ ] Update ratings from win, draw, or loss only. Do not use goal margin in the rating update.
- [ ] Start with ordinary Elo as the transparent baseline.
- [ ] Add a TrueSkill-style rating with mean `mu` and uncertainty `sigma` once the match pipeline is trusted.
- [ ] Report a conservative score such as `mu - 3 * sigma` alongside the mean rating.
- [ ] Match challengers mostly against similarly rated checkpoints, with regular games against fixed anchor policies.
- [ ] Keep random, scripted, built-in, and early trained policies as permanent anchors to detect ladder drift.
- [ ] Require matches against several distinct opponents before a checkpoint can be promoted.
- [ ] Run periodic round-robin evaluation among the strongest checkpoints to detect non-transitive strategies.
- [ ] Publish rating history, uncertainty, win rates by opponent, and replay links in a local report.

Completion check: rebuilding the ladder from the stored match records produces identical ratings, and a promoted checkpoint beats the incumbent across paired fixtures rather than only showing a higher training return.

## Later, after the baseline works

- [ ] Compare IPPO with MAPPO and a centralized critic.
- [ ] Add partial observations and test decentralized execution.
- [ ] Compare one fully shared actor, role-specific heads, and separate goalkeeper weights.
- [ ] Test recurrent actors after feed-forward policies and reset handling are stable.
- [ ] Investigate a simulator with native futsal pitch dimensions and rules if GRF's approximation changes the research question.
- [ ] Add distributed rollout workers only when local profiling shows simulation throughput is the bottleneck.
