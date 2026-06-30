# Workspace restoration

This repository includes a full research snapshot through Git LFS: datasets,
research evidence, generated reports, model artifacts, and original source
material.

## Restore on another computer

1. Install Git and Git LFS.
2. Clone the repository.
3. Download every LFS object:

   ```powershell
   git lfs install
   git lfs pull
   ```

4. Create a Python environment and install `requirements.txt`.
5. Compare restored files with
   `migration/manifests/full_workspace_sha256.csv` when exact integrity matters.

Python bytecode, pytest caches, virtual environments, local secrets, and the
Git object database are intentionally excluded. They are machine-specific or
reconstructible and are not research evidence.
