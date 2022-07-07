cd ../src/

python main.py tracking --exp_id wildpark --dataset custom --custom_dataset_ann_path ../data/WildparkDataset/annotations/test.json --custom_dataset_img_path ../data/WildparkDataset/annotated_frames --input_h 1296 --input_w 1728 --num_classes 2 --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --save_all --test --load_model ../exp/tracking/MobileNetV2/model_last.pth --arch generic --backbone mobilenet