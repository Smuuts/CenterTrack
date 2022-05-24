
cd ../src/
python main.py tracking --exp_id wildpark --dataset custom --custom_dataset_ann_path ../data/WildparkDataset/annotations/train.json --custom_dataset_img_path ../data/WildparkDataset/annotated_frames --input_h 1296 --input_w 1728 --num_classes 2 --num_epochs 1 --batch_size 1 --arch dla_34 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0
# --input_h 1296 --input_w 1728 funktioniert nicht
# python main.py tracking --exp_id wildpark --dataset custom --custom_dataset_ann_path ../data/WildparkDataset/annotations/train.json --custom_dataset_img_path ../data/WildparkDataset/annotated_frames --input_h 1088 --input_w 1920 --num_classes 2 --num_epochs 1 --batch_size 1 --arch dla_34 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0
# --input_h 1088 --input_w 1920 funktioniert