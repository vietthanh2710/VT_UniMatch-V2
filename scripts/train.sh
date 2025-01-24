dataset='doc_semi'
method='unimatch_v2'
exp='dinov2_small'
split='366'

config=configs/${dataset}.yaml
# labeled_id_path=/kaggle/input/doc-data/2_DocumentSegmentation/train.txt
labeled_id_path=/kaggle/input/doc-path-no-tamtru/train.txt
unlabeled_id_path=/kaggle/input/doc3d-image/Doc3d_dataset/file_path.txt
save_path=/kaggle/working/exp/semi

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log
