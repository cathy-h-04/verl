# Metric Inventory By File Type
## `experiment_name.txt`
- Files found: **11**
- Unique metric/key paths: **1**

| Metric key path | Representative value |
|---|---|
| `value` | `stage2_llama8b_grpo_20260301_122602` |

## `slurm_job_ids.txt`
- Files found: **11**
- Unique metric/key paths: **4**

| Metric key path | Representative value |
|---|---|
| `slurm_array_task_id` | `3` |
| `slurm_job_id` | `63031634` |
| `slurm_job_name` | `stage2_llama_check` |
| `timestamp` | `20260301_123547` |

## `slurm_config.json`
- Files found: **11**
- Unique metric/key paths: **10**

| Metric key path | Representative value |
|---|---|
| `cpus_per_task` | `16` |
| `error` | `/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.err` |
| `gpus_per_node` | `4` |
| `job_name` | `stage2_llama_check` |
| `mem` | `48G` |
| `nodes` | `1` |
| `ntasks_per_node` | `1` |
| `output` | `/n/netscratch/yu_lab/Lab/chou/logs/%A_%a.out` |
| `partition` | `gpu_h200` |
| `time` | `00:45:00` |

## `run_config.json`
- Files found: **11**
- Unique metric/key paths: **29**

| Metric key path | Representative value |
|---|---|
| `meta.index` | `3` |
| `meta.name` | `stage2_llama8b_grpo` |
| `meta.source` | `/n/home08/chou/verl_research/profiling_scripts/experiments/stage2_llama8b/runs.json` |
| `run.dataset` | `gsm8k` |
| `run.gpus_per_node` | `4` |
| `run.granularity` | `operation` |
| `run.model` | `meta-llama/Llama-3.1-8B-Instruct` |
| `run.name` | `stage2_llama8b_grpo` |
| `run.nnodes` | `1` |
| `run.policy` | `grpo` |
| `run.poll_interval` | `0.5` |
| `run.resume_path` | `/n/netscratch/yu_lab/Lab/chou/checkpoints/stage1_llama8b_grpo_20260226_192746/global_step_50` |
| `run.rollout_n` | `2` |
| `run.save_freq` | `-1` |
| `run.total_epochs` | `1` |
| `run.total_steps` | `null` |
| `run.use_validation` | `true` |
| `run.val_freq` | `10` |
| `run.val_max_samples` | `128` |
| `train.enable_grad_checkpointing` | `true` |
| `train.gpu_memory_util` | `0.5` |
| `train.log_prob_micro_batch_size` | `4` |
| `train.micro_batch_size_per_gpu` | `4` |
| `train.ppo_mini_batch_size` | `32` |
| `train.rollout_max_batched_tokens` | `8192` |
| `train.rollout_max_model_len` | `2048` |
| `train.rollout_max_num_seqs` | `64` |
| `train.tensor_parallel_size` | `1` |
| `train.train_batch_size` | `128` |

## `nvml_boundary.jsonl`
- Files found: **11**
- Unique metric/key paths: **52**

| Metric key path | Representative value |
|---|---|
| `clocks_throttle_reasons_raw` | `0` |
| `driver_version` | `575.57.08` |
| `elapsed_seconds` | `186.208355595` |
| `error_fields` | `["temp_mem_C"]` |
| `gpu_energy_mJ` | `1998577127362` |
| `gpu_enforced_power_limit_mW` | `700000` |
| `gpu_index` | `0` |
| `gpu_name` | `NVIDIA H200` |
| `gpu_power_limit_mW` | `700000` |
| `gpu_power_mW` | `112763` |
| `gpu_util_pct` | `0` |
| `gpu_uuid` | `GPU-c7c8dd50-92a7-0c3e-1c3b-665ef6955545` |
| `graphics_clock_MHz` | `1980` |
| `iteration` | `51` |
| `local_rank` | `0` |
| `mem_clock_MHz` | `3201` |
| `mem_free_B` | `135517437952` |
| `mem_total_B` | `150754820096` |
| `mem_used_B` | `15237382144` |
| `mem_util_pct` | `0` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `pcie_link_gen` | `5` |
| `pcie_link_width` | `16` |
| `pcie_rx_bytes_s` | `593920` |
| `pcie_tx_bytes_s` | `546816` |
| `phase_duration_s` | `11.161296465` |
| `phase_event` | `START` |
| `phase_gpu_energy_delta_J` | `2661.339` |
| `phase_id` | `1` |
| `phase_name` | `rollout` |
| `pid` | `3157616` |
| `pstate` | `0` |
| `rank` | `0` |
| `record_type` | `PHASE_BOUNDARY` |
| `sm_clock_MHz` | `1980` |
| `sm_util_pct` | `0` |
| `temp_gpu_C` | `29` |
| `temp_mem_C` | `null` |
| `thr_apps_clocks_setting` | `false` |
| `thr_display_clock_setting` | `false` |
| `thr_hw_power_brake` | `false` |
| `thr_hw_slowdown` | `false` |
| `thr_hw_thermal_slowdown` | `false` |
| `thr_idle` | `false` |
| `thr_sw_power_cap` | `false` |
| `thr_sw_thermal_slowdown` | `false` |
| `thr_sync_boost` | `false` |
| `thr_thermal_slowdown` | `false` |
| `timestamp` | `2026-03-01_12:31:17` |
| `ts_monotonic_ns` | `4591021167286794` |
| `ts_wall_ms` | `1772386277186` |
| `world_size` | `1` |

## `nvml_periodic.jsonl`
- Files found: **11**
- Unique metric/key paths: **49**

| Metric key path | Representative value |
|---|---|
| `clocks_throttle_reasons_raw` | `1` |
| `driver_version` | `575.57.08` |
| `elapsed_seconds` | `0.353519312` |
| `error_fields` | `["temp_mem_C"]` |
| `gpu_energy_mJ` | `1998556276061` |
| `gpu_enforced_power_limit_mW` | `700000` |
| `gpu_index` | `0` |
| `gpu_name` | `NVIDIA H200` |
| `gpu_power_limit_mW` | `700000` |
| `gpu_power_mW` | `74057` |
| `gpu_util_pct` | `0` |
| `gpu_uuid` | `GPU-c7c8dd50-92a7-0c3e-1c3b-665ef6955545` |
| `graphics_clock_MHz` | `345` |
| `iteration` | `0` |
| `local_rank` | `0` |
| `mem_clock_MHz` | `3201` |
| `mem_free_B` | `150106013696` |
| `mem_total_B` | `150754820096` |
| `mem_used_B` | `648806400` |
| `mem_util_pct` | `0` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `pcie_link_gen` | `5` |
| `pcie_link_width` | `16` |
| `pcie_rx_bytes_s` | `666624` |
| `pcie_tx_bytes_s` | `701440` |
| `phase_id` | `0` |
| `phase_name` | `idle` |
| `pid` | `3157616` |
| `pstate` | `0` |
| `rank` | `0` |
| `record_type` | `PERIODIC` |
| `sm_clock_MHz` | `345` |
| `sm_util_pct` | `0` |
| `temp_gpu_C` | `27` |
| `temp_mem_C` | `null` |
| `thr_apps_clocks_setting` | `false` |
| `thr_display_clock_setting` | `false` |
| `thr_hw_power_brake` | `false` |
| `thr_hw_slowdown` | `false` |
| `thr_hw_thermal_slowdown` | `false` |
| `thr_idle` | `true` |
| `thr_sw_power_cap` | `false` |
| `thr_sw_thermal_slowdown` | `false` |
| `thr_sync_boost` | `false` |
| `thr_thermal_slowdown` | `false` |
| `timestamp` | `2026-03-01_12:28:11` |
| `ts_monotonic_ns` | `4590835312454891` |
| `ts_wall_ms` | `1772386091331` |
| `world_size` | `1` |

## `rapl_boundary.jsonl`
- Files found: **11**
- Unique metric/key paths: **21**

| Metric key path | Representative value |
|---|---|
| `cpu_energy_uJ` | `258178172192` |
| `domain_path` | `/sys/class/powercap/intel-rapl:0` |
| `elapsed_seconds` | `186.347575547` |
| `error_fields` | `[]` |
| `iteration` | `51` |
| `local_rank` | `0` |
| `max_energy_range_uJ` | `262143328850` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `phase_domain_energy_delta_uJ` | `2492038847` |
| `phase_duration_s` | `11.352087504` |
| `phase_event` | `START` |
| `phase_id` | `1` |
| `phase_name` | `rollout` |
| `pid` | `3157616` |
| `rank` | `0` |
| `rapl_domain` | `package-0` |
| `record_type` | `PHASE_BOUNDARY` |
| `timestamp` | `2026-03-01_12:31:17` |
| `ts_monotonic_ns` | `4591021306519321` |
| `ts_wall_ms` | `1772386277325` |
| `world_size` | `1` |

## `rapl_periodic.jsonl`
- Files found: **11**
- Unique metric/key paths: **26**

| Metric key path | Representative value |
|---|---|
| `cpu_energy_uJ` | `217460927185` |
| `cpu_freq_max_MHz` | `2910.211` |
| `cpu_freq_mean_MHz` | `2900.354205357143` |
| `cpu_freq_min_MHz` | `2900.0` |
| `cpu_util_pct_total` | `null` |
| `domain_path` | `/sys/class/powercap/intel-rapl:0` |
| `elapsed_seconds` | `1.880157523` |
| `error_fields` | `[]` |
| `iteration` | `0` |
| `load1` | `2.28` |
| `load15` | `4.0` |
| `load5` | `3.34` |
| `local_rank` | `0` |
| `max_energy_range_uJ` | `262143328850` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `phase_id` | `0` |
| `phase_name` | `idle` |
| `pid` | `3157616` |
| `rank` | `0` |
| `rapl_domain` | `package-0` |
| `record_type` | `PERIODIC` |
| `rss_bytes_process` | `1123909632` |
| `timestamp` | `2026-03-01_12:28:12` |
| `ts_monotonic_ns` | `4590836839096632` |
| `ts_wall_ms` | `1772386092858` |
| `world_size` | `1` |

## `phase_timings_[EXP_NAME].jsonl`
- Files found: **11**
- Unique metric/key paths: **15**

| Metric key path | Representative value |
|---|---|
| `elapsed_seconds` | `197.508445752` |
| `error_fields` | `[]` |
| `iteration` | `51` |
| `local_rank` | `0` |
| `metric_unit` | `s` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `phase_id` | `1` |
| `phase_name` | `rollout` |
| `pid` | `3157616` |
| `rank` | `0` |
| `subphase_name` | `comm_s/gen` |
| `ts_monotonic_ns` | `4591032467391009` |
| `ts_wall_ms` | `1772386288486` |
| `value` | `0.0` |
| `world_size` | `1` |

## `[EXP_NAME].jsonl`
- Files found: **11**
- Unique metric/key paths: **85**

| Metric key path | Representative value |
|---|---|
| `data.actor/entropy` | `0.36684396862983704` |
| `data.actor/grad_norm` | `0.513671875` |
| `data.actor/lr` | `1e-06` |
| `data.actor/pg_clipfrac` | `2.9204429665696807e-05` |
| `data.actor/pg_clipfrac_lower` | `0.0` |
| `data.actor/pg_loss` | `0.0013312011687958147` |
| `data.actor/ppo_kl` | `1.9908864601347886e-05` |
| `data.comm_fraction/step` | `0.0014879652103211287` |
| `data.comm_s/gen` | `0.0` |
| `data.comm_s/gen_max` | `0.0` |
| `data.comm_s/old_log_prob` | `0.04347490519285202` |
| `data.comm_s/step` | `0.04523430671542883` |
| `data.comm_s/update_actor` | `0.04523430671542883` |
| `data.comm_s/update_critic` | `0.022795445285737514` |
| `data.comm_s/values` | `0.02106127329170704` |
| `data.critic/advantages/max` | `0.7071057558059692` |
| `data.critic/advantages/mean` | `-0.012121553532779217` |
| `data.critic/advantages/min` | `-0.7071057558059692` |
| `data.critic/grad_norm` | `32.265625` |
| `data.critic/lr` | `1e-05` |
| `data.critic/returns/max` | `0.7071057558059692` |
| `data.critic/returns/mean` | `-0.012121553532779217` |
| `data.critic/returns/min` | `-0.7071057558059692` |
| `data.critic/rewards/max` | `1.0` |
| `data.critic/rewards/mean` | `0.91015625` |
| `data.critic/rewards/min` | `0.0` |
| `data.critic/score/max` | `1.0` |
| `data.critic/score/mean` | `0.91015625` |
| `data.critic/score/min` | `0.0` |
| `data.critic/values/max` | `1.0078125` |
| `data.critic/values/mean` | `0.7578125` |
| `data.critic/values/min` | `0.158203125` |
| `data.critic/vf_clipfrac` | `0.0` |
| `data.critic/vf_explained_var` | `-0.049215078353881836` |
| `data.critic/vf_loss` | `0.02628546178675606` |
| `data.critic/vpred_mean` | `0.9186952225863934` |
| `data.global_seqlen/balanced_max` | `18490` |
| `data.global_seqlen/balanced_min` | `18186` |
| `data.global_seqlen/max` | `18373` |
| `data.global_seqlen/mean` | `18262.75` |
| `data.global_seqlen/min` | `18153` |
| `data.global_seqlen/minmax_diff` | `220` |
| `data.logging/granularity` | `operation` |
| `data.logging/record_scope` | `iteration_summary` |
| `data.logging/validation_logged` | `false` |
| `data.logging/wall_time` | `1772386307.53569` |
| `data.perf/cpu_memory_used_gb` | `36.65617370605469` |
| `data.perf/max_memory_allocated_gb` | `91.34893846511841` |
| `data.perf/max_memory_reserved_gb` | `102.068359375` |
| `data.perf/mfu/actor` | `0.06557848960687729` |
| `data.perf/mfu/critic` | `0.0658624262025163` |
| `data.perf/throughput` | `600.7461729378817` |
| `data.perf/total_num_tokens` | `73051` |
| `data.prompt_length/clip_ratio` | `0.0` |
| `data.prompt_length/max` | `191.0` |
| `data.prompt_length/mean` | `102.1484375` |
| `data.prompt_length/min` | `69.0` |
| `data.response/aborted_ratio` | `0.0` |
| `data.response_length/clip_ratio` | `0.0` |
| `data.response_length/max` | `942.0` |
| `data.response_length/mean` | `183.20703125` |
| `data.response_length/min` | `76.0` |
| `data.response_length_non_aborted/clip_ratio` | `0.0` |
| `data.response_length_non_aborted/max` | `942.0` |
| `data.response_length_non_aborted/mean` | `183.20703125` |
| `data.response_length_non_aborted/min` | `76.0` |
| `data.rollout/response_len_bucket_256_768` | `24` |
| `data.rollout/response_len_bucket_gt_768` | `1` |
| `data.rollout/response_len_bucket_lt_256` | `231` |
| `data.rollout/response_length_p50` | `171.0` |
| `data.rollout/response_length_p95` | `290.5` |
| `data.rollout/response_length_std` | `79.83209991455078` |
| `data.rollout/straggler_ratio` | `5.141724109649658` |
| `data.rollout/sync_efficiency` | `0.1944872885942459` |
| `data.timing_per_token_ms/adv` | `0.00011281752089543783` |
| `data.timing_per_token_ms/gen` | `0.23253700233834992` |
| `data.timing_per_token_ms/update_actor` | `0.1881358429778936` |
| `data.timing_per_token_ms/update_critic` | `0.1875074253334242` |
| `data.timing_per_token_ms/values` | `0.0474879304061335` |
| `data.timing_s/gen` | `10.90621794667095` |
| `data.timing_s/step` | `30.40011043380946` |
| `data.training/epoch` | `0` |
| `data.training/global_step` | `51` |
| `data.val-core/openai/gsm8k/reward/mean@1` | `0.875` |
| `step` | `51` |

## `tokens_and_steps.jsonl`
- Files found: **11**
- Unique metric/key paths: **27**

| Metric key path | Representative value |
|---|---|
| `elapsed_seconds` | `197.302210047` |
| `error_fields` | `[]` |
| `iteration` | `51` |
| `local_rank` | `0` |
| `metric_scope` | `tokens_and_steps` |
| `node` | `holygpu8a12202.rc.fas.harvard.edu` |
| `phase_id` | `1` |
| `phase_name` | `rollout` |
| `pid` | `3157616` |
| `rank` | `0` |
| `record_type` | `PERIODIC` |
| `rollout_max_seq_len` | `942` |
| `rollout_mean_output_len` | `183.20703125` |
| `rollout_num_sequences` | `256` |
| `rollout_output_tokens_total` | `46901` |
| `rollout_prompt_tokens_total` | `26150` |
| `rollout_total_tokens` | `73051` |
| `timestamp` | `2026-03-01_12:31:28` |
| `train_batch_tokens` | `73051` |
| `train_epochs` | `1` |
| `train_microbatch_tokens_estimated` | `1141` |
| `train_minibatch_count_estimated` | `8` |
| `train_minibatch_passes_estimated` | `1` |
| `train_tokens_effective_estimated` | `73051` |
| `ts_monotonic_ns` | `4591032261146626` |
| `ts_wall_ms` | `1772386288280` |
| `world_size` | `1` |
