cd data/reddit/source/

python3 get_raw_users.py

echo 'Good job with raw'

python3 merge_raw_users.py

echo 'Good job with merging'

python3 clean_raw.py

echo 'Good job with cleaning'

python3 delete_small_users.py

echo 'Good job subsampling'

python3 get_json.py

echo 'Good job creating json'

python3 preprocess.py

echo 'Good job preprocessing'
