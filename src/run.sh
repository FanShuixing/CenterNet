#训练coco
# python main.py ctdet --exp_id coco_dla --batch_size 16 --master_batch 15 --lr 1.25e-4  --gpus 0

#训练自定义数据
# python main.py ctdet --exp_id pascal_dla_384 --dataset pascal --num_epochs 70 --lr_step 45,60 --batch_size 24

#predict 有gt
#python demo.py ctdet --demo val_mask.csv --load_model /output/tf_dir/ctdet/pascal_dla_384/model_best.pth

#predict 无gt
python predict.py ctdet --demo val_mask.csv --load_model /output/tf_dir/ctdet/pascal_dla_384/model_best.pth

#计算F1 score
# python pascalvoc.py  --gtfolder /input0/ --detfolder ./ --threshold 0.7
