CUDA_VISIBLE_DEVICES=4 python ./inference.py \
/home/lake/Project/olaz/dataset/public_test_2/videos \
--num-gpu=1 \
--model=swin_base_patch4_window7_224 \
--num-classes=2 \
--checkpoint /home/lake/Project/olaz/pytorch-image-models/output/train/20221109-224757-swin_base_patch4_window7_224-224/last.pth.tar