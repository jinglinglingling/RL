# OSWorld third_party backup patches

These patches capture the local recovery fixes from third_party/OSWorld without pushing upstream history that triggers secret-scanning on GitHub.

## Included commits
- 28d6a2ba
- 8d48e973

Apply with:

```bash
git am 0001-apptainer-provider-stability.patch
git am 0002-rollout-context-overflow-and-fallback.patch
```
