python utils/data_split.py \
--report_files_dir /home/e1373616/project/sph6004/physionet.org/files/mimic-cxr-jpg/2.1.0/files \
--split_path /home/e1373616/project/sph6004/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz \
--out_dir /home/e1373616/project/sph6004/Group_work/dataset/

# get .h5 data for embedding training
python dataset/emb_dataset.py
## train
python train_img_emb.py > result/train_img_emb_stdout.txt 2> result/train_img_emb_stderr.txt