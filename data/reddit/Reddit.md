# Creating Reddit dataset

If you want to download an already processed dataset, refer to the [Readme](../Readme.md).

Parsing and word tokenization process is followed as in [Leaf](https://github.com/TalwalkarLab/leaf). 

1. Download the file **`RC_2017-12.zst`** from [Reddit comments](https://files.pushshift.io/reddit/comments/), unpack it and place it into `data/reddit/` (use [zstd](https://github.com/facebook/zstd) to unpack)
2. Run `sh run_reddit.sh`
3. In `data/reddit/preparation.py,` run the function `create_tokenizer` with the desired vocabulary size from the desired generated Reddit jsons
4. In `data/reddit/preparation.py,` run the function `parse_reddit_file` with the index of the generated Reddit json file you wish to create processed datasets from. This will generate multiple `.h5` files containing data of maximum `max_client_num` users. To increase sequence length, set `word_backwards` to the desired value.
