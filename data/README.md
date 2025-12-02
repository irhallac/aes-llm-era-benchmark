# Data Directory

This repository does not distribute the Feedback Prize datasets or precomputed
embeddings. Create this layout locally before running any experiment:

```
data/
  train_set_kaggle.csv
  stratified_100.csv
  stratified_500.csv
  embeddings/
    <*.npz /.npy files generated via data/scripts>
  scripts/
    <helpers for building embeddings>
```

See `docs/embeddings.md` for details on generating embeddings. All files under
`data/` are ignored by git so they remain private on your machine.
