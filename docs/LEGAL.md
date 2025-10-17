# Legal & Repository Hygiene

Do not distribute ROMs. This repository must never contain `.nes` files or any dumped ROM content.

Git ignore
- `*.nes` and `legal_ROMs/` are ignored in `.gitignore`.

Pre-commit guard
1. Make the hook executable: `chmod +x .git-hooks/pre-commit-block-roms.sh`
2. Enable it: `ln -sf ../../.git-hooks/pre-commit-block-roms.sh .git/hooks/pre-commit`
3. The hook blocks committing `.nes` files or `legal_ROMs/` paths.

Purge any accidental history (if already committed)
Note: This removes files from Git history but not your local filesystem.

```
git rm -rf --cached legal_ROMs
git commit -m "Remove ROMs from repo"
pip install git-filter-repo
git filter-repo --path legal_ROMs --path-glob '*.nes' --invert-paths
git push --force --all
git push --force --tags
```

Always keep ROMs outside the repository and reference their location in local configs only.
