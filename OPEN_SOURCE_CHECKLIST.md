# Open Source Release Checklist

Before publishing this repository on GitHub:

- Choose and add a license file, for example `LICENSE`.
- Confirm that the compact exported `state_dict` checkpoints under `checkpoints/` can be redistributed publicly.
- Verify that any datasets referenced by examples are public or documented as external dependencies.
- Run the example scripts after replacing local paths with paths available to users.
- Confirm the paper citation and revised project title are final.
- Review generated outputs and large binary files before committing.

This release copy intentionally includes compact exported `state_dict` checkpoints for reviewer assessment, but excludes original full-object checkpoints, generated reconstructions, cached files, and local datasets.
