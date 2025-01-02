dataset='doc'
method='supervised_weighted'
exp='doc'
split='366'

config=configs/${dataset}.yaml
labeled_id_path=/kaggle/input/doc-data/2_DocumentSegmentation/train.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=/kaggle/working/exp/doc/supervised

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log
