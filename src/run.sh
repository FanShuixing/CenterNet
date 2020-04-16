#训练
python main.py ctdet --exp_id coco_dla --batch_size 16 --master_batch 15 --lr 1.25e-4  --gpus 0

# python main.py ctdet --exp_id pascal_dla_384 --dataset pascal --num_epochs 70 --lr_step 45,60

#predict
# python demo.py ctdet --demo /input0/test2017/000000097335.jpg --load_model /output/ctdet_coco_dla_2x.pth
