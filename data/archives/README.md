# X-NERF++ Dataset Archive Drop Zone

Put compressed dataset files here on your PC before uploading them as a Kaggle Dataset.

Expected locations:

```text
data/archives/
  malnet_tiny/
    images/
      *.zip | *.tar | *.tar.gz | *.tgz
    graphs/
      *.zip | *.tar | *.tar.gz | *.tgz
  AndMal2020/
    static/
      *.zip | *.tar | *.tar.gz | *.tgz
    dynamic/
      *.zip | *.tar | *.tar.gz | *.tgz
  cicmaldroid2020/
    *.zip | *.tar | *.tar.gz | *.tgz
  drebin/
    *.zip | *.tar | *.tar.gz | *.tgz
  ember/
    *.zip | *.tar | *.tar.gz | *.tgz
  virusshare/
    *.zip | *.tar | *.tar.gz | *.tgz
  cape/
    *.zip | *.tar | *.tar.gz | *.tgz
```

On Kaggle, upload this `data/archives` folder as a Kaggle Dataset named something like `xnerf-malware-archives`.

The notebook script expects archives at:

```text
/kaggle/input/xnerf-malware-archives/archives/<dataset_name>/
```

If Kaggle nests the upload one level differently, set `XNERF_KAGGLE_ARCHIVE_ROOT` in the notebook before running materialization.

Use `data/archives` for compressed input files. The extracted working copy is created separately under `data/raw` or `/kaggle/working/data/raw`.

You can add future datasets with subdirectories without changing code:

```text
data/archives/<new_dataset_name>/<modality_or_split>/*.zip
```

Example:

```text
data/archives/my_new_dataset/static/features.zip
data/archives/my_new_dataset/dynamic/traces.tar.gz
```

The extractor writes them to:

```text
data/raw/my_new_dataset/static/
data/raw/my_new_dataset/dynamic/
```
