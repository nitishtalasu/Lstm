## CS512 Final Project Code

### Instructions to run code

You can execute `sh main.sh` to run the entire pipeline, with the required arguments already specified in this file.

##### Notes:
- A sample yelp (short) dataset for the VAE is given in datasets/. 
- Provide the labeled documents. For example, if the dataset is `yelp`, replace the contents of `datasets/yelp/short_yelp.test.txt` with the labeled documents.
- To ensure folder structure is consistent (while creating datasets and models), please run all the following steps from the parent directory
- Essential changes to code can be found in 
    - Interpolation and document generation: `vae/modules/vae.py`
    - Classification of pseudo docs: `westclass/gen.py` 

##### Steps:
1. Follow the instructions given in `vae/README.md` under "Usage" to train the VAE model. Models will be saved under exp_abcd/

2. Now that the VAE model is trained (and labeled documents are given), run
```
python text_anneal_fb.py \
    --dataset ${dataset} \
    --reconstruct_from ${model_path_from_step_1} \
    --decoding_strategy sample
```
to generate the pseudo documents from the labeled documents provided in step 3. These will be written into the file `pseudo_documents.txt`

5. Run 
```
python3 westclass/main.py --dataset ${dataset} --sup_source docs --model cnn --with_evaluation True
```
to obtain the final results.

- Code borrowed from https://github.com/bohanli/vae-pretraining-encoder and https://github.com/yumeng5/WeSTClass

