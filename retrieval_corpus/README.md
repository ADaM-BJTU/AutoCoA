## Corpus Preparation


Please first ensure that your working directory is set to the current folder, and then you can install the dependencies by running:

```bash
pip install -e .
```

We completely follow [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)'s setup to prepare the external environmental corpus needed for the retrieval stage. You can download and view the 2021 Wikipedia corpus that I have segmented [here](https://modelscope.cn/datasets/yatzteng/AutoCOA_wikicorpus21_256w). Afterwards, you can use `scripts/build_index.sh` to create the corresponding line index file. In our experimental setup, we chose `intfloat/e5-base-v2` as our vector encoding model and used faiss as the tool for index construction.

Then start the retrieval service locally in API form by running `python -m scripts.run_retrieval_server`.
