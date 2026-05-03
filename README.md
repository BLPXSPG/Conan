# Conan: Official Code for Understanding Complex Relationships in Detective Narratives

This repository contains the official code and benchmark resources for the paper **[Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives](https://aclanthology.org/2024.findings-acl.454/)**, published in **Findings of ACL 2024**.

Conan is a benchmark for narrative understanding, character relationship extraction, and relation graph analysis in detective narratives. The dataset focuses on complex social relationships, role-oriented perspectives, public and secret character relations, and long-context reasoning for large language models.

**Keywords:** Conan benchmark, detective narratives, narrative understanding, character relationship extraction, relation graph, social relation extraction, long-context reasoning, role-oriented annotation, LLM evaluation.

- Paper: [ACL Anthology](https://aclanthology.org/2024.findings-acl.454/)
- PDF: [2024.findings-acl.454.pdf](https://aclanthology.org/2024.findings-acl.454.pdf)


## Environment
Dependency packages are in `environment.yml`, you can update the prefix. You can then create a new environment from this file:

```{bash}
conda env create -f environment.yml
```

## Data
The preprocessed data included in `[data/]` is available in two languages, English and Chinese. The original data is in Chinese, with story names as folder names in `[data/chinese/data_final]`. Access to the original version requires purchase.

Each narrative features descriptions from various characters' perspectives, located in `[data/language/data_final/narrative_name/txt/character_name]`. The corresponding labels of relationships from a single character's perspective can be found in `[data/language/label/narrative_name/txt/character_name]`, and the relationships combining information from all characters are in `[data/language/label/narrative_name/txt/all]`.

![Input-Output Format and Benchmark Relation Detection Strategies](conan-framework.jpeg)


## Citation
If you use the Conan dataset or benchmark in your research, please cite our paper:

Runcong Zhao, Qinglin Zhu, Hainiu Xu, Jiazheng Li, Yuxiang Zhou, Yulan He, and Lin Gui. 2024. [Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives](https://aclanthology.org/2024.findings-acl.454/). In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 7618-7638, Bangkok, Thailand. Association for Computational Linguistics.

```bibtex
@inproceedings{zhao-2024-conan,
  title = {Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives},
  author = {Zhao, Runcong and Zhu, Qinglin and Xu, Hainiu and Li, Jiazheng and Zhou, Yuxiang and He, Yulan and Gui, Lin},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages = {7618--7638},
  year = {2024},
  address = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2024.findings-acl.454},
  url = {https://aclanthology.org/2024.findings-acl.454/}
}
```
