<h1 align="center">
<img src="docs/images/embodied-logo.png" alt="embodied-logo" width="40" height="40" style="vertical-align: middle; margin-top: -12px;">
EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents
</h1>

---

# Stage-2 Submission — Team233

This fork is Team233's Stage-2 submission to the CVPR 2026 FMEA EmbodiedBench Challenge. It uses **Option A + B**: two fine-tuned Qwen3.5-VL-9B checkpoints served via vLLM plus a customized planner under `embodiedbench/planner/`. Everything outside `embodiedbench/planner/` is the unmodified upstream code (evaluator, environments, configs, metrics).

## Models (Option A)

Two fine-tuned 9B checkpoints, one per task. Each is served separately.

| Task | Served name | Description | Download |
|---|---|---|---|
| EB-ALFRED | `Qwen3.5-Alf-SFT` | Qwen3.5-VL-9B + LoRA SFT on successful EB-ALFRED multi-step trajectories, merged | TODO: add Hugging Face / Drive link |
| EB-Navigation | `Qwen3.5-Nav-RL` | Qwen3.5-VL-9B + Unsloth-GRPO (initialized from the base 9B) on replayed Nav states, merged | TODO: add Hugging Face / Drive link |

The exact served names (`Qwen3.5-Alf-SFT`, `Qwen3.5-Nav-RL`) **must be preserved**: the planner uses substring matching on the served name to (a) route through the OpenAI-compatible vLLM client in `remote_model.py` and (b) swap in the Stage-2 system prompts inside `vlm_planner.py` / `nav_planner.py`.

## Planner customizations (Option B)

All changes vs. upstream live under `embodiedbench/planner/`:

- `embodiedbench/planner/custom_prompts.py` *(new)* — Stage-2 ALFRED and Nav system prompts plus the model-name detectors. The planners overwrite `self.system_prompt` here at construction time so the upstream `embodiedbench/evaluator/config/system_prompts.py` stays byte-identical to upstream.
- `embodiedbench/planner/vlm_planner.py` — adds plan-length capping, an action validator that drops repeated invalid pick-ups when the agent is already holding an object, dict-based history feedback with `task_progress` / `holding`, and the Stage-2 prompt override (when the served model is `Qwen3.5-Alf-SFT`, also overrides `n_shot=5`).
- `embodiedbench/planner/nav_planner.py` — adds distance-change feedback in the action history, plan-length capping, a fallback action on JSON/empty errors, an action validator that drops repeated failed/regressing moves, and the Stage-2 prompt override.
- `embodiedbench/planner/remote_model.py` — adds a `Qwen3.5` branch that talks to a local vLLM OpenAI-compatible endpoint (`remote_url` env var) with `temperature=1.0` and `max_completion_tokens=8192`.

The planner relies only on fields already provided by the upstream environments (`info['distance']` in `EBNavEnv.py`, `info['task_progress']` / `info['last_action_success']` / `info['action_description']` in `EBAlfEnv.py`).

## How to run

### 1. Install the upstream EmbodiedBench environments

Follow the standard upstream install: a single conda env per environment, then `pip install -e .` inside each. For Stage-2 only the ALFRED and Navigation envs are needed.

```bash
conda env create -f conda_envs/environment.yaml         # creates `embench` (EB-ALFRED)
conda activate embench && pip install -e .

conda env create -f conda_envs/environment_eb-nav.yaml  # creates `embench_nav` (EB-Navigation)
conda activate embench_nav && pip install -e .
```

EB-ALFRED dataset / EB-Habitat / EB-Manipulation steps from `install.sh` are unchanged. Only the ALFRED dataset clone (`git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED ...`) is required for our submission.

### 2. Create a separate env for vLLM

Our checkpoints are Qwen3.5-VL-9B (Qwen-thinking disabled). The `embench` env is not built for vLLM serving, so create a small dedicated env and install vLLM following its own instructions (https://docs.vllm.ai/en/latest/getting_started/installation.html). A minimal setup:

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm           # follow the vLLM install guide for your CUDA/driver
```

### 3. Run the evaluation

The submission requires **two separate runs** — one per task — because each task uses its own model. Both runs share the same calling convention: launch vLLM on port 8000, then run `embodiedbench.main` from the corresponding conda env, pointing `remote_url` at the vLLM endpoint.

#### 3a. EB-ALFRED

```bash
# Terminal A — vLLM server
conda activate vllm
vllm serve /path/to/Qwen3.5-Alf-SFT \
  --served-model-name Qwen3.5-Alf-SFT \
  --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 32768

# Terminal B — evaluator
conda activate embench
export remote_url=http://127.0.0.1:8000/v1
python -m embodiedbench.main \
  env=eb-alf \
  model_name=Qwen3.5-Alf-SFT \
  model_type=remote \
  exp_name=stage2_alf
```

#### 3b. EB-Navigation

```bash
# Terminal A — vLLM server (replace the ALFRED one)
conda activate vllm
vllm serve /path/to/Qwen3.5-Nav-RL \
  --served-model-name Qwen3.5-Nav-RL \
  --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 32768

# Terminal B — evaluator
conda activate embench_nav
export remote_url=http://127.0.0.1:8000/v1
python -m embodiedbench.main \
  env=eb-nav \
  model_name=Qwen3.5-Nav-RL \
  model_type=remote \
  exp_name=stage2_nav
```

The required vLLM flags are:
- `--reasoning-parser qwen3` and `--default-chat-template-kwargs '{"enable_thinking": false}'` — disables Qwen3's thinking trace, which is necessary because our training data is non-thinking JSON.
- `--max-model-len 32768` — needed because the planner prompts (with history and few-shot examples) can be longer than the default context.

The served-model-name and the evaluator's `model_name` must match exactly (`Qwen3.5-Alf-SFT` or `Qwen3.5-Nav-RL`); the planner uses substring matching on this name to apply Stage-2-specific behavior.

---


<p align="center">
  <span style="color:red;">
    EmbodiedBench will be used for 
    <a href="https://foundation-models-meet-embodied-agents.github.io/cvpr2026/"><strong>CVPR 2026 Workshop on Foundation Models Meet Embodied Agents (FMEA) Challenges</strong></a>!
    Stay tuned.
  </span>
</p>


<p align="center">
  📄  <a href="http://arxiv.org/abs/2502.09560"><strong>Paper</strong></a> |  
  🤗 <a href="https://huggingface.co/EmbodiedBench"><strong>Dataset</strong></a> |
  🏠 <a href="https://embodiedbench.github.io"><strong>Project Website</strong></a>
</p>


<p align="center">
    <a href="https://yangrui2015.github.io/">Rui Yang*</a>, 
    <a href="">Hanyang Chen*</a>, 
    <a href="https://jyzhang1208.github.io/">Junyu Zhang*</a>, 
    <a href="">Mark Zhao*</a>, 
    <a href="https://qiancheng0.github.io/">Cheng Qian</a>, 
    <a href="https://jameskrw.github.io/">Kangrui Wang</a>, 
    <a href="https://qinengwang-aiden.github.io/">Qineng Wang</a>, 
    <a href="">Teja Venkat Koripella</a>, 
    <a href="">Marziyeh Movahedi</a>, 
    <a href="https://limanling.github.io/">Manling Li</a>, 
    <a href="https://blender.cs.illinois.edu/hengji.html">Heng Ji</a>, 
    <a href="https://www.huan-zhang.com/">Huan Zhang</a>, 
    <a href="https://tongzhang-ml.org/">Tong Zhang</a>
</p>
<p align="center">University of Illinois Urbana-Champaign, Northwestern University, University of Toronto, Toyota Technological Institute at Chicago</p>


<img src="docs/images/framework.png" width="100%" />

# 🔥 Overview 
We introduce **EmbodiedBench**, a comprehensive benchmark designed to evaluate **Multi-modal Large Language Models (MLLMs) as embodied agents**. While existing benchmarks have primarily focused on Large Language Models (LLMs) and high-level tasks, EmbodiedBench takes a leap forward by offering a comprehensive, fine-grained evaluation of MLLM-based agents across both **high-level and low-level tasks**, as well as **six critical agent capabilities**. 

EmbodiedBench is more than a benchmark—it’s a **multifaceted, standardized evaluation platform** that not only uncovers the current challenges in embodied AI but also provides actionable insights to push the boundaries of MLLM-driven embodied agents.  

## 📌 News
- 2025.10 We released [Embodied Reasoning Agent (ERA)](https://arxiv.org/abs/2510.12693), a training recipe for VLM-based embodied agents with enhanced reasoning and grounding capability. Explore more on our [project page](https://embodied-reasoning-agent.github.io/)!
- 2025.06.03 We released a large collection of [trajectory datasets](https://huggingface.co/EmbodiedBench) generated by a diverse set of models, including both closed-source and open-source models. Feel free to use them to train better embodied agents!
- 2025.05.01 EmbodiedBench is accepted to **ICML 2025**!
- 2025.03.19, we provided support for several recent MLLMs, including "microsoft/Phi-4-multimodal-instruct", 'AIDC-AI/Ovis2-16B', 'AIDC-AI/Ovis2-34B', 'google/gemma-3-12b-it', and fixed some common generated JSON errors.


## 🚀 **Key Features** 

- 🛠️ **Diverse Tasks with Hierarchical Action Levels:**  
  **1,128 testing tasks** across four environments, spanning from high-level tasks (EB-ALFRED and EB-Habitat) to low-level tasks (EB-Navigation and EB-Manipulation). We created new high-quality datasets and enhanced existing simulators to support comprehensive assessments.

- 🎯 **Capability-Oriented Evaluation:**  
  **Six specialized subsets** to evaluate essential agent capabilities, including commonsense reasoning, complex instruction, spatial awareness, visual perception, and long-term planning. 

- ⚡ **Unified APIs for Embodied Environments:**  
  EmbodiedBench provides **[Gym](https://github.com/openai/gym)-style APIs for all environments**, ensuring ease of use and seamless agent evaluation.  

- 🏹 **Effortless MLLM/LLM Evaluation (API & Local Support):**  
  - Supports **proprietary** (e.g., OpenAI API) and **open-source models** (local execution).  
  - Enables self-hosted model evaluation using OpenAI API-style calls or offline execution based on [LMDeploy](https://github.com/InternLM/lmdeploy).  
  - While mainly focused on MLLMs, EmbodiedBench also supports **LLM evaluation**.  


- 🔧 **Configurable Textual and Visual Designs:**  
Our flexible configuration options enable in-depth experimentation with **visual input**, **textual and visual in-context prompts**, **environment feedback**, **camera resolution**, **detection boxes**, and **multi-step/multi-view image inputs** and more, empowering researchers to better understand the role of each component in agent performance.


### Planning Examples in EB-ALFRED and EB-Manipulation
<img src="docs/images/planning_example.png" width="100%" />


### Comparison with related benchmarks

"Fine-grained" indicates a multi-dimensional evaluation approach rather than an overall accuracy.  
¹AgentBench and VisualAgentBench include domains such as household, games, and web.  
²VLABench is originally used for evaluating VLA models.

| Benchmark | Category | Action Level | #Env. | #Test Tasks | Multimodal | Fine-grained | LLM/VLM Support |
|-----------|----------|--------------|------|-------------|------------|--------------|-----------------|
| Alfred  | Household | High | 1 | 3062 | ✅ | ❌ | ❌ |
| VLMbench  | Manipulation | Low | 1 | 4760 | ✅ | ❌ | ❌ |
| Language Rearrangement  | Household | High | 1 | 1000 | ✅ | ✅ | ❌ |
| GOAT-bench  | Navigation | Low | 1 | 3919 | ✅ | ❌ | ❌ |
| AgentBench  | Multi-domain¹ | High | 8 | 1091 | ❌ | ❌ | ✅ |
| Lota-bench  | Household | High | 2 | 308 | ❌ | ❌ | ✅ |
| VisualAgentBench  | Multi-domain¹ | High | 5 | 746 | ✅ | ❌ | ✅ |
| Embodied Agent Interface  | Household | High | 2 | 438 | ❌ | ✅ | ✅ |
| VLABench  | Manipulation | Low² | 1 | 100 | ✅ | ✅ | ✅ |
| **EmbodiedBench (ours)** | Multi-domain | High & Low | 4 | 1128 | ✅ | ✅ | ✅ |



# 🖥️ Installation
**Note: we need to install three conda environments, one for EB-ALFRED and EB-Habitat, one for EB-Navigation, and one for EB-Manipulation. Please use ssh download instead of HTTP download to avoid error during git lfs pull.**

Download repo
```bash
git clone git@github.com:EmbodiedBench/EmbodiedBench.git
cd EmbodiedBench
```

**You have two options for installation: you can either use 
```bash install.sh``` or manually run the provided commands. After completing the installation with `bash install.sh`, you will need to start the headless server and verify that each environment is properly set up.**

1️⃣ Environment for ```Habitat and Alfred```
```bash
conda env create -f conda_envs/environment.yaml 
conda activate embench
pip install -e .
```
2️⃣ Environment for ```EB-Navigation```
```bash
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate embench_nav
pip install -e .
```
3️⃣ Environment for ```EB-Manipulation```
```bash
conda env create -f conda_envs/environment_eb-man.yaml 
conda activate embench_man
pip install -e .
```

**Note: EB-Alfred, EB-Habitat and EB-Manipulation require downloading large datasets from Hugging Face or GitHub repositories. Ensure Git LFS is properly initialized by running the following commands:**
```bash
git lfs install
git lfs pull
```

## Start Headless Server
Please run startx.py script before running experiment on headless servers. The server should be started in another tmux window. We use X_DISPLAY id=1 by default.
```bash
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
```

## EB-Alfred
Download dataset from huggingface.
```bash
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```
Run the following code to ensure the EB-ALFRED environment is working correctly. `Remember to start headless server.`

```bash
conda activate embench
python -m embodiedbench.envs.eb_alfred.EBAlfEnv
```

## EB-Habitat

> Update: the default long-horizon EB-Habitat dataset has been updated in place. Please use `embodiedbench/envs/eb_habitat/datasets/long_horizon.pickle`. The previous version is preserved as `embodiedbench/envs/eb_habitat/datasets/long_horizon_old.pickle`. This fix is related to [Issue #28](https://github.com/EmbodiedBench/EmbodiedBench/issues/28) and [Issue #39](https://github.com/EmbodiedBench/EmbodiedBench/issues/39).

- Install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) via

 ```bash
conda activate embench
conda install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat
git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..
 ```

- Download YCB and ReplicaCAD dataset for the Language Rearrangement task. 
```bash
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat
```
After the above step, there should be a data folder under envs/eb_habitat.

Run the following code to ensure the EB-Habitat environment is working correctly.
```bash
conda activate embench
python -m embodiedbench.envs.eb_habitat.EBHabEnv
```

## EB-Navigation

Run the following code to ensure the EB-Navigation environment is working correctly.
```bash
conda activate embench_nav
python -m embodiedbench.envs.eb_navigation.EBNavEnv
```

## EB-Manipulation
* Install Coppelia Simulator

CoppeliaSim V4.1.0 required for Ubuntu 20.04; you can find other versions here (https://www.coppeliarobotics.com/previousVersions#)

```bash
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ /PATH/YOU/WANT/TO/PLACE/COPPELIASIM
```

* Add the following to your *~/.bashrc* file:

```bash
export COPPELIASIM_ROOT=/PATH/YOU/WANT/TO/PLACE/COPPELIASIM
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

> Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

* Install the PyRep, EB-Manipulation package and dataset:
```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation
mv EB-Manipulation/data/ ./
rm -rf EB-Manipulation/
cd ../../..
```

> Remember that whenever you re-install the PyRep, simAddOnScript_PyRep.lua will be overwritten. Then, you should copy this again.

* Run the following code to ensure the EB-Manipulation is working correctly (start headless server if you have not):
```bash
conda activate embench_man
export DISPLAY=:1
python -m embodiedbench.envs.eb_manipulation.EBManEnv
```


# 🚀 Quick Start
### Proprietary Models
Before running evaluations, set up your environment variables if you plan to use proprietary models:
```bash
export OPENAI_API_KEY="your_oai_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"
export ANTHROPIC_API_KEY="your_anpic_api_key_here"
export DASHSCOPE_API_KEY="your_dashscope_api_here" # the official qwen apis
```
To evaluate MLLMs in EmbodiedBench, activate the corresponding Conda environment and run:
```bash
conda activate embench
python -m embodiedbench.main env=eb-alf model_name=gpt-4o-mini exp_name='baseline'
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name='baseline'

conda activate embench_nav
python -m embodiedbench.main env=eb-nav model_name=gpt-4o exp_name='baseline'

conda activate embench_man 
python -m embodiedbench.main env=eb-man model_name=claude-3-5-sonnet-20241022 exp_name='baseline'
```
#### Configuration Options
You can customize the evaluation using the following flags:
- **`env`**: The environment to test. Choose from:  
  - `'eb-alf'` (EB-ALFREd)  
  - `'eb-hab'` (EB-Habitat)  
  - `'eb-man'` (EB-Manipulation)  
  - `'eb-nav'` (EB-Navigation)  

- **`model_name`**: Full model name, including proprietary options like:  
  - `'gpt-4o'`, `'gpt-4o-mini'`, `'claude-3-5-sonnet-20241022'`, `'gemini-1.5-pro'`, `'gemini-2.0-flash-exp'`, `'gemini-1.5-flash'`  

- **`model_type`**: Set to `'remote'` by default.  
- **`down_sample_ratio`**: Data sampling ratio (default `1.0`). Use `0.1` for debugging (10% of the dataset).  
- **`language_only`**: If `True` (or `1`), the agent receives only text input (default: `False`).  
- **`eval_sets`**: List of subsets to evaluate (default: all subsets).  
- **`chat_history`**: Enables multi-turn interaction (`False` by default, as it may reduce performance).  
- **`n_shots`**: Maximum number of textual examples for in-context learning (varies by environment).  
- **`multiview`**: Uses multi-view images as input (only for EB-Manipulation & EB-Navigation, default: `False`).  
- **`multistep`**: Includes historical multi-step images (`False` by default).  
- **`detection_box`**: Enables detection box input (valid for EB-ALFREd, EB-Navigation, and EB-Manipulation).  
- **`resolution`**: Image resolution (default: `500`).  
- **`exp_name`**: Name of the experiment, used in logging.  
- **`visual_icl`**: Enables visual in-context learning (`False` by default).  
- **`log_level`**: Sets the logging level (`INFO` by default). Use `DEBUG` for debugging purposes.
- **`truncate`**: **[Now only for EB-Navigation since other tasks normally don't require chat_history=True]** Enables truncation of conversation history when `chat_history=True` (`False` by default). When enabled, it automatically removes verbose content from previous conversation turns while preserving key information. Only takes effect when `chat_history=True`.

> ⚠️ **Important:** Avoid enabling multiple flags simultaneously from `visual_icl`, `multiview`, `multistep`, and `chat_history` to prevent excessive image inputs and conflicts.  

#### More on "truncate" for EB-Navigation

**🔧 Context Management with Truncate:**  
For long navigation tasks with `chat_history=True`, the conversation history can become quite lengthy, potentially affecting model performance and exceeding context limits. The `truncate` feature addresses this by preprocessing the message history and truncating repetitive prompts before sending it to the model.

The reason why there might be unnecessary but lengthy prompt is that we wish to support a "WINDOW_SIZE"(which can be set in the corresponding planner.py file) argument when chat_history is set to True. The window will select the last "WINDOW_SIZE" messages before sending to the model. Therefore, we first allow each message to contain a system prompt and then truncate all of them except the last message. This way, we avoid the case that there will be no system prompt when the message length exceeds "WINDOW_SIZE", if the system prompt is only included in the first message.

**Usage Example:**
```bash
# Enable chat history with truncation for better context management
conda activate embench_nav
python -m embodiedbench.main env=eb-nav model_name=gpt-4o chat_history=True truncate=True exp_name='nav_with_truncation'

# Compare with standard chat history (without truncation)
python -m embodiedbench.main env=eb-nav model_name=gpt-4o chat_history=True truncate=False exp_name='nav_standard_history'

# Standard evaluation without chat history (truncate has no effect)
python -m embodiedbench.main env=eb-nav model_name=gpt-4o chat_history=False exp_name='nav_no_history'
```

**When to use `truncate=True`:**
- Long navigation episodes (>10 steps) with `chat_history=True`
- Models with limited context windows
- When experiencing performance degradation due to overly long conversation history
- To reduce API costs for proprietary models by managing token usage

---

### Open-source Models
We support two deployment methods for open-source models: **offline running** and **model serving**.  

#### **1️⃣ Offline Running**  
For local execution, set `model_type=local` and adjust `tp` (tensor parallelism) based on GPU memory.  
- A rough guideline: For 48GB GPUs, use `tp = ceil(model size (in B) / 10)`.  
```bash
conda activate embench
python -m embodiedbench.main env=eb-alf model_name=Qwen/Qwen2-VL-7B-Instruct model_type=local exp_name='baseline' tp=1
python -m embodiedbench.main env=eb-hab model_name=OpenGVLab/InternVL2_5-8B model_type=local exp_name='baseline' tp=1


conda activate embench_nav
python -m embodiedbench.main env=eb-nav model_name=OpenGVLab/InternVL2_5-38B model_type=local exp_name='baseline' tp=4

conda activate embench_man 
python -m embodiedbench.main env=eb-man model_name=meta-llama/Llama-3.2-11B-Vision-Instruct model_type=local exp_name='baseline' tp=2
```

#### **2️⃣ Online Model Serving (Recommended)**  
Model serving decouples **model execution** from **evaluation**, allowing flexible deployment via API calls.  
```bash
## Step 0, create an environment for lmdeploy
conda env create -f conda_envs/lmdeploy.yaml
conda activate lmdeploy
pip install lmdeploy

## Step 1, open another tmux window, runing the model
lmdeploy serve api_server "OpenGVLab/InternVL2_5-8B" --server-port $port --tp 1

## Step 2, running the evaluation
conda activate embench
export remote_url='IP_address:port/v1' # set the address for access, e.g., http://localhost:8000.
python -m embodiedbench.main env=eb-hab model_name=OpenGVLab/InternVL2_5-8B exp_name='baseline' 
```
You can also refer to [LMDeploy](https://github.com/InternLM/lmdeploy) for more details.


#### **3️⃣ Online Serving for Unsupported Models**  
Lmdeploy often lags behind the release of new models. To address this, we offer a more flexible and dynamic model serving approach. Follow these steps to deploy and evaluate new models:

```bash
## 1. Modify the code and hyperparameters in `server.py` according to your requirements.
## We now support "microsoft/Phi-4-multimodal-instruct", 'AIDC-AI/Ovis2-16B', 'AIDC-AI/Ovis2-34B', 'google/gemma-3-12b-it' 
## 2. Start the server and install any necessary packages:
pip install flask
CUDA_VISIBLE_DEVICES=${gpu_ids} python server.py

## 3. Run the evaluation in custom mode:
export server_url="IP_address:port/process"
python -m embodiedbench.main env=eb-hab model_name='microsoft/Phi-4-multimodal-instruct' model_type='custom' exp_name='new_model'
```


## Docker
We have provided a Docker file under the Docker folder. 



# Acknowledgement
This repo is based on awesome embodied benchmarks and simulations [Lota-Bench](https://github.com/lbaa2022/LLMTaskPlanning), [ALFRED](https://github.com/askforalfred/alfred), [ai2thor](https://github.com/allenai/ai2thor), [EmbodiedAgentInterface](https://github.com/embodied-agent-interface/embodied-agent-interface), [ML-Llarp](https://github.com/apple/ml-llarp), [Habitat](https://github.com/facebookresearch/habitat-lab), [VLMBench](https://github.com/eric-ai-lab/VLMbench), and [RLBench](https://github.com/stepjam/RLBench). Our open-source model deployment is based on [LMDeploy](https://github.com/InternLM/lmdeploy) and [vllm](https://github.com/vllm-project/vllm). 

# Citation
```
@inproceedings{
yang2025embodiedbench,
title={EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents},
author={Rui Yang and Hanyang Chen and Junyu Zhang and Mark Zhao and Cheng Qian and Kangrui Wang and Qineng Wang and Teja Venkat Koripella and Marziyeh Movahedi and Manling Li and Heng Ji and Huan Zhang and Tong Zhang},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=DgGF2LEBPS}
}
```
