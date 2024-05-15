# Deep Hierarchy-aware Proxy Hashing with Self-paced Learning for Cross-modal Retrieval [Paper](https://ieeexplore.ieee.org/document/10530441)
This paper is accepted for publication with TKDE.

## Dependencies

- pytorch 1.12.1
- sklearn
- tqdm
- pillow

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128


### Citation
@ARTICLE{10530441,  
  author={Huo, Yadong and Qin, Qibing and Zhang, Wenfeng and Huang, Lei and Nie, Jie},  
  journal={IEEE Transactions on Knowledge and Data Engineering},  
  title={Deep Hierarchy-aware Proxy Hashing with Self-paced Learning for Cross-modal Retrieval},  
  year={2024},  
  volume={},  
  number={},  
  pages={1-14},  
  doi={10.1109/TKDE.2024.3401050}}


### Acknowledgements
[DCHMT](https://github.com/kalenforn/DCHMT)
