from animals_dataset import AnimalDatasetImagecsv
import json

DATA_PATH = '../../data/WildparkDataset/'
OUT_PATH = DATA_PATH + 'annotations/'
IMG_PATH = DATA_PATH + 'annotated_frames/'

SPLITS = ['train', 'test']

if __name__ == '__main__':
    data = AnimalDatasetImagecsv(IMG_PATH, DATA_PATH + 'pre_csv.csv', 'Wildpark', preprocessed=True)
    ann_cnt = 0
    for split in SPLITS:        # train and test split
        out_path = OUT_PATH + '{}.json'.format(split)

        out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'red deer'}, {'id': 2, 'name': 'roe deer'}],
           'videos': [{'id': 3, 'file_name': 'DW_2020_11_10_VD_00051'}, {'id': 8, 'file_name': 'RW_2020_11_10_VD_00151'}]}
           
        for i in range(len(data)):      # iterate through amount of videos
            imgs = data[i]
            video_length = len(imgs)    
            split_index = SPLITS.index(split)

            imgs_per_split = int(video_length/len(SPLITS))
            
            for j in range(split_index * imgs_per_split, (split_index + 1) * imgs_per_split):   # give every split equal part of video
                frame = data.getframe(imgs[j])[1]
                print(f'Converting {imgs[j]}')

                img_id = int(frame['image_id'].item())
                image_info = {'file_name': imgs[j],
                      'id': img_id,
                      'frame_id': img_id,
                      'prev_image_id': img_id - 1 if (img_id + 1)/(i+1) > 0 else -1,
                      'next_image_id': (img_id + 1) if (img_id + 1)/(i+1) < video_length else -1,
                      'video_id': -1}

                for video in out['videos']:
                    if video['file_name'] in imgs[j]:
                        image_info['video_id'] = video['id']
                        break

                out['images'].append(image_info)

                for k in range(len(frame['boxes'])):
                    ann = {'id': ann_cnt,
                        'category_id': frame['labels'][k].item(),
                        'image_id': img_id,
                        'track_id': frame['track'][k].item(),
                        'bbox': frame['boxes'][k].tolist(),
                        'conf': 1.0}

                    ann_cnt = ann_cnt + 1                    

                    out['annotations'].append(ann)

        print(f'writing {split} split to {out_path}')
        json.dump(out, open(out_path, 'w'))
    print('done!')