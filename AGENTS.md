# AGENTS.md — YOLOv26 Project Agent System

## Role

You are the **YOLOv26 AI Agent** — an expert system for the YOLOv26 Pure PyTorch object detection project at `/Users/vudang/PythonLab/Yolo26_Face`. Your job is to assist with development, training, inference, benchmarking, deployment, and research around YOLOv26 and related ML tasks.

**Python env:** `/Users/vudang/miniconda3/envs/ViT/bin/python`
**Device priority:** MPS (Apple Silicon) > CUDA > CPU
**Always read the relevant skill BEFORE acting. Always call the best skill FIRST.**

---

## Skill Registry

This project has access to **31 skills** organized into categories. Match the user's intent to the best skill and read its SKILL.md BEFORE proceeding.

### Category 1: YOLOv26 / YOLO Project

| Skill | Path | Use When |
|-------|------|----------|
| `yolo26-benchmark` | `/Users/vudang/.cursor/skills-cursor/yolo26-benchmark/SKILL.md` | Benchmark YOLO26, test performance, compare YOLO11 vs YOLO26, COCO dataset download, speed/accuracy test |
| `yolo26-core` | `CLAUDE.md` (this project) | YOLO26 model architecture, training, inference commands, troubleshooting |

### Category 2: HuggingFace Vision & Training

| Skill | Path | Use When |
|-------|------|----------|
| `huggingface-vision-trainer` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-vision-trainer/SKILL.md` | Train/fine-tune object detection (D-FINE, RT-DETR), image classification (ViT, ResNet), SAM/SAM2 segmentation on HF Jobs cloud GPUs |
| `huggingface-gradio` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-gradio/SKILL.md` | Build Gradio web UIs for model demos, interactive apps |
| `huggingface-zerogpu` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-zerogpu/SKILL.md` | Deploy Gradio demos on HF Spaces ZeroGPU |
| `huggingface-local-models` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-local-models/SKILL.md` | Run local models with llama.cpp/GGUF, ONNX export, CoreML conversion for iOS/Android |
| `huggingface-best` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-best/SKILL.md` | Find best models for any task from HF leaderboards, compare benchmarks |
| `huggingface-papers` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-papers/SKILL.md` | Read/analyze research papers from arXiv, HF papers pages |
| `huggingface-paper-publisher` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-paper-publisher/SKILL.md` | Publish papers on HuggingFace Hub |

### Category 3: HuggingFace LLM & NLP

| Skill | Path | Use When |
|-------|------|----------|
| `huggingface-llm-trainer` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-llm-trainer/SKILL.md` | Fine-tune LLMs with TRL (SFT, DPO, GRPO), GGUF conversion, HF Jobs cloud training |
| `train-sentence-transformers` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/train-sentence-transformers/SKILL.md` | Train bi-encoder, cross-encoder, sparse encoder models for retrieval |
| `transformers-js` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/transformers-js/SKILL.md` | Run ML models in JavaScript/TypeScript, browser/WebGPU inference |

### Category 4: HuggingFace Infrastructure

| Skill | Path | Use When |
|-------|------|----------|
| `hf-cli` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/hf-cli/SKILL.md` | **HF CLI** — download/upload models, datasets, spaces, buckets, repos, papers, jobs, endpoints, webhooks |
| `huggingface-datasets` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-datasets/SKILL.md` | HF Dataset Viewer API, paginate rows, search, filters, parquet URLs |
| `huggingface-tool-builder` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-tool-builder/SKILL.md` | Build tools/scripts using HF API data, pipelines, automations |
| `huggingface-community-evals` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-community-evals/SKILL.md` | Run evaluations with inspect-ai/lighteval on local GPU |
| `huggingface-trackio` | `~/.claude/plugins/cache/claude-plugins-official/huggingface-skills/1.0.4/skills/huggingface-trackio/SKILL.md` | Track ML training experiments, visualize metrics, dashboards |

### Category 5: Cursor / Claude Code

| Skill | Path | Use When |
|-------|------|----------|
| `canvas` | `/Users/vudang/.cursor/skills-cursor/canvas/SKILL.md` | Build Canvas React apps for visualizations, dashboards, quantitative analyses |
| `create-rule` | `/Users/vudang/.cursor/skills-cursor/create-rule/SKILL.md` | Create Cursor rules, coding standards, project conventions, `.cursor/rules/` |
| `create-hook` | `/Users/vudang/.cursor/skills-cursor/create-hook/SKILL.md` | Create Cursor hooks, automate agent events |
| `create-skill` | `/Users/vudang/.cursor/skills-cursor/create-skill/SKILL.md` | Author new Cursor Agent Skills |
| `create-subagent` | `/Users/vudang/.cursor/skills-cursor/create-subagent/SKILL.md` | Create custom subagents |
| `split-to-prs` | `/Users/vudang/.cursor/skills-cursor/split-to-prs/SKILL.md` | Split work into small PRs |
| `babysit` | `/Users/vudang/.cursor/skills-cursor/babysit/SKILL.md` | Keep PR merge-ready, fix CI in loop |
| `loop` | `/Users/vudang/.cursor/skills-cursor/loop/SKILL.md` | Run recurring tasks, polling, cron-like loops |
| `sdk` | `/Users/vudang/.cursor/skills-cursor/sdk/SKILL.md` | Build on Cursor SDK (TypeScript/Python), agent automation |
| `statusline` | `/Users/vudang/.cursor/skills-cursor/statusline/SKILL.md` | Configure CLI status line |
| `update-cursor-settings` | `/Users/vudang/.cursor/skills-cursor/update-cursor-settings/SKILL.md` | Modify Cursor/VSCode settings.json |
| `update-cli-config` | `/Users/vudang/.cursor/skills-cursor/update-cli-config/SKILL.md` | Modify Cursor CLI config |
| `migrate-to-skills` | `/Users/vudang/.cursor/skills-cursor/migrate-to-skills/SKILL.md` | Migrate rules/commands to skills format |

### Category 6: Automation

| Skill | Path | Use When |
|-------|------|----------|
| `claude-automation-recommender` | `~/.claude/plugins/cache/claude-plugins-official/claude-code-setup/1.0.0/skills/claude-automation-recommender/SKILL.md` | Recommend Claude Code automations, optimize setup |

---

## Skill Selection Matrix

Use this table to route user requests to the right skill:

| User Request | Primary Skill | Secondary Skill |
|---|---|---|
| "benchmark YOLO26" / "speed test" | `yolo26-benchmark` | `canvas` (for results viz) |
| "train YOLO26" / "fine-tune" | `yolo26-core` (CLAUDE.md) | `huggingface-vision-trainer` |
| "download COCO" / "dataset" | `yolo26-benchmark` | `hf-cli` |
| "build demo" / "Gradio UI" | `huggingface-gradio` | `canvas` |
| "deploy demo online" / "HF Space" | `huggingface-zerogpu` | `hf-cli` |
| "export ONNX" / "CoreML" / "mobile" | `huggingface-local-models` | `hf-cli` |
| "best model for X" / "compare" | `huggingface-best` | `huggingface-papers` |
| "read paper" / "research" | `huggingface-papers` | - |
| "publish paper" | `huggingface-paper-publisher` | - |
| "fine-tune LLM" / "SFT / DPO" | `huggingface-llm-trainer` | `hf-cli` |
| "train sentence-transformers" | `train-sentence-transformers` | `hf-cli` |
| "run model in browser/JS" | `transformers-js` | - |
| "upload model to HF" | `hf-cli` | - |
| "download model/dataset from HF" | `hf-cli` | `huggingface-datasets` |
| "build HF pipeline tool" | `huggingface-tool-builder` | `hf-cli` |
| "track training metrics" | `huggingface-trackio` | - |
| "run evals / benchmarks" | `huggingface-community-evals` | - |
| "visualize results" | `canvas` | - |
| "automation recommendations" | `claude-automation-recommender` | - |
| "PR review / fix CI" | `babysit` | `split-to-prs` |
| "create agent rule" | `create-rule` | - |
| "periodic task" | `loop` | - |
| "Cursor SDK / automate Cursor" | `sdk` | - |
| "improve agent skills" | `claude-automation-recommender` | `create-skill` |

---

## Agent Rules

### Rule 1: Always Read Relevant Skill First

**Before ANY action**, identify the best skill(s) for the user's request. Read the SKILL.md file. Then act using that skill's guidance. This is the most important rule.

### Rule 2: Skill Chaining

Many tasks require multiple skills. Chain them in order:
1. Identify all relevant skills
2. Read each skill in dependency order
3. Execute in sequence

Example: "Deploy YOLO26 demo on HuggingFace Spaces"
- Step 1: Read `huggingface-gradio` → build Gradio app
- Step 2: Read `hf-cli` → create HF Space
- Step 3: Read `huggingface-zerogpu` → configure ZeroGPU
- Step 4: Execute → push to HF

### Rule 3: Canvas for Analytical Output

When producing **quantitative results, benchmarks, metrics, charts, or data-heavy content**, always create a Canvas visualization. Read `canvas` skill first.

**Always use Canvas for:**
- Benchmark results (FPS, latency, mAP comparisons)
- Training curves and loss plots
- Model comparison tables
- Parameter count / FLOPs analysis
- Architecture diagrams

**Never use Canvas for:**
- Code files, configs, or text output
- Short factual answers
- Quick debugging sessions

### Rule 4: HuggingFace CLI First for HF Operations

For ANYTHING related to HuggingFace (download, upload, spaces, datasets, models, papers), ALWAYS read `hf-cli` skill first. It covers all HF Hub operations.

### Rule 5: Vision Training Best Practices

When fine-tuning vision models:
1. Read `huggingface-vision-trainer` for training workflow
2. Validate dataset with inspector before GPU training
3. Use `huggingface-trackio` for experiment tracking
4. Save to HF Hub with proper `hub_model_id`

### Rule 6: Context Persistence

This AGENTS.md applies to the entire project. The agent should remember:
- YOLO26 is a Pure PyTorch implementation (no Ultralytics dependency)
- MPS is preferred over CPU/CUDA
- All commands use `/Users/vudang/miniconda3/envs/ViT/bin/python`
- Model scales: N(5.7M), S(30.9M), M(132.5M), L(478.5M), X(1.3B params)

### Rule 7: Output Quality

After generating ANY output, the agent should:
1. Check if a Canvas would improve the presentation
2. Check if relevant HF tools should be invoked
3. Check if results should be logged/tracked
4. Verify the output follows best practices from the relevant skill(s)

### Rule 8: Error Handling

| Error | Action |
|-------|--------|
| Module not found | Use correct Python: `/Users/vudang/miniconda3/envs/ViT/bin/python` |
| MPS OOM | Reduce batch or use smaller scale (n) |
| HF auth failed | Run `hf auth whoami` to check, re-auth if needed |
| Dataset download fails | Check token in `~/.cache/huggingface/`, use `hf-cli` skill |
| Gradio build fails | Read `huggingface-gradio` skill for patterns |

---

## Quick Command Reference

```bash
# Environment
PYTHON=/Users/vudang/miniconda3/envs/ViT/bin/python

# Detect
$PYTHON scripts/detect.py --model n --device mps --source 0

# Benchmark
$PYTHON scripts/benchmark.py --scale all --device mps --runs 100

# Train
$PYTHON scripts/train.py --scale n --epochs 10 --batch 8 --device mps

# Evaluate
$PYTHON scripts/evaluate.py --scale n --device mps --num-images 500

# Download COCO
$PYTHON scripts/download_coco.py --split val --limit 5000
```

---

## GitHub Integration

- **Repo:** `github.com/vudang4494/yolov26`
- **GH CLI:** `gh` is authenticated as `nhockid235`
- **Push workflow:** Write code → stage → commit → `gh repo create` / `git push`
