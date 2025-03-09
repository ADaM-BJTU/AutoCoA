python preprocess_wiki.py \
    --dump_path /fs-ift/nlp/yangyuqi/projects/coa/wikipedia_corpus/enwiki-20210120-pages-articles/enwiki-20210120-pages-articles.xml.bz2  \
    --save_path /fs-ift/nlp/yangyuqi/projects/coa/wikipedia_corpus/output_corpus_256w.jsonl \
    --chunk_by word \
    --chunk_size 256 \
    --num_workers 32
    # --tokenizer_name_or_path word \