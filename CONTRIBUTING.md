# Contributing & Git Workflow

## Branch Strategy

We use a simple **feature-branch workflow**:

```
main              ← stable, always runnable
│
├── dev           ← integration branch (merge here first)
│   │
│   ├── feat/data-pipeline       ← Person A
│   ├── feat/vision-encoder      ← Person B
│   ├── feat/text-encoder        ← Person B
│   ├── feat/infonce-loss        ← Person A
│   ├── feat/training-loop       ← Person B
│   ├── feat/evaluation          ← Person A
│   └── feat/gradio-demo         ← Person B
```

**Rule:** Never commit directly to `main`. Always go through a feature branch → PR into `dev` → final merge into `main` at end of week.

---

## Daily Workflow

```bash
# Start of day — sync with latest
git checkout dev
git pull origin dev

# Create or switch to your feature branch
git checkout feat/your-feature
# or
git checkout -b feat/new-feature

# Work, then commit often with clear messages
git add src/dataset/flickr_dataset.py
git commit -m "feat(dataset): add random caption sampling per epoch"

# Push to remote
git push origin feat/your-feature

# When feature is complete → open PR into dev on GitHub
```

---

## Commit Message Format

Use this format for clean history:

```
<type>(<scope>): <short description>

Types:
  feat      New feature
  fix       Bug fix
  docs      Documentation only
  refactor  Code restructure (no behavior change)
  exp       Experiment / ablation run
  chore     Build, config, dependencies
```

**Examples:**
```
feat(loss): implement symmetric InfoNCE with learnable temperature
fix(dataset): handle missing captions in results.csv
docs(arch): add projection MLP diagram to ARCHITECTURE.md
exp(ablation): add temperature=0.05 run results
refactor(encoders): move L2 normalization into ProjectionMLP forward()
```

---

## Pull Request Checklist

Before opening a PR into `dev`, confirm:

- [ ] Code runs without errors
- [ ] New functions have docstrings
- [ ] Any new hyperparameters are added to `config.py`
- [ ] Notebook outputs are cleared before committing (`Kernel > Restart & Clear Output`)
- [ ] No data files accidentally staged (check `git status`)

---

## Weekly Sync Points

| Day | Action |
|-----|--------|
| Monday | Merge all week-1 feature branches into `dev`, review together |
| Wednesday | Mid-week check-in — unblock each other |
| Friday | Merge `dev` → `main` with a version tag (`v0.1-training-done`) |
| End of project | Final tag: `v1.0-submission` |

---

## Handling Conflicts

If you get a merge conflict:

```bash
git checkout dev
git pull origin dev
git checkout feat/your-feature
git merge dev           # brings dev changes into your branch
# Resolve conflicts in editor
git add .
git commit -m "chore: resolve merge conflict with dev"
```

---

## Shared Notebook Etiquette

Notebooks are a common source of conflicts since they store outputs as JSON.

**Rules:**
1. Only one person edits a notebook at a time
2. Always clear outputs before committing: `Kernel → Restart & Clear Output`
3. Prefer `.py` scripts for reusable code — notebooks are for exploration and presentation only
4. Name your exploration notebooks with initials: `03_training_AB.ipynb`
