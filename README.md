## Environment
Dependency packages are in environment.yml, you can update the prefix. You can then create a new environment from this file:

```{bash}
conda env create -f environment.yml
```

## Data
The preprocessed data included in `[data/]` is available in two languages, English and Chinese. The original data is in Chinese, with story names as folder names in `[data/chinese/data_final]`. Access to the original version requires purchase.

Each narrative features descriptions from various characters' perspectives, located in `[data/language/data_final/narrative_name/txt/character_name]`. The corresponding labels of relationships from a single character's perspective can be found in `[data/language/label/narrative_name/txt/character_name]`, and the relationships combining information from all characters are in `[data/language/label/narrative_name/txt/all]`.

