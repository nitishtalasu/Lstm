### Files information

1. `short_yelp.train.txt`,  `short_yelp.valid.txt` and `short_yelp.test.txt` are used during VAE training
2. `dataset.csv` and `doc_id.txt` are used during westclass pre- and self-training
3. `doc_id.txt` contains IDs of the labeled documents chosen from the training data
4. `dataset.csv` and `short_yelp.train.txt` essentially are the same training data, with the difference being the labels (0,1 as opposed to 1,2). 
5. The difference in labels between the two files is only for yelp data as we did not want to modify the westclass code too much. The labels will be the same in case of ag_news (0-3) for both VAE and westclass training
6. Create a folder `agnews` with similar files to train on the AG News dataset