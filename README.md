# Model-Compression
Model compression for green AI 

The configuration
Baseline
"python scripts/train_benchmark_resnet50_cifar10_ACCURATE.py \
  --pretrained --device cpu \
  --img_size 128 \
  --epochs 8 \
  --batch_size 32 \
  --num_workers 2 \
  --finetune staged \
  --head_warmup_epochs 2 \
  --layer4_epochs 6 \
  --optimizer sgd \
  --lr 0.1 \
  --backbone_lr 0.01 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --warmup_epochs 1 \
  --randaugment \
  --track_energy \
  --output_dir outputs_run2"
///////////////////////////////////
Knowledge Distillation
ython scripts/train_student_kd_cifar10.py \
  --teacher_ckpt outputs/resnet50_cifar10_best.pt \
  --student resnet18 \
  --img_size 128 \
  --epochs 15 \
  --batch_size 64 \
  --num_workers 2 \
  --alpha 0.7 \
  --temperature 4.0 \
  --lr 0.05 \
  --weight_decay 1e-4 \
  --train_aug \
  --track_energy \
  --output_dir outputs_kd_student
////////////////////////////////////
Searchin for the best prun structure
python scripts/search_prune_quant_kd.py \
  --student_ckpt outputs_kd_student/student_best.pt \
  --teacher_ckpt outputs/resnet50_cifar10_best.pt \
  --student resnet18 \
  --img_size 128 \
  --prune_scopes layer4 layer3_layer4 \
  --prune_amounts 0.1 0.2 0.3 0.4 \
  --quant_modes int8 \
  --kd_eval_batches 50 \
  --calib_batches 200 \
  --output_dir outputs_search
///////////////////////////////////
prune + short KD finetune (3 epochs)
python scripts/prune_student_structured.py \
  --student_ckpt outputs_kd_student/student_best.pt \
  --teacher_ckpt outputs/resnet50_cifar10_best.pt \
  --student resnet18 \
  --img_size 128 \
  --prune_scope layer3_layer4 \
  --prune_amount 0.3 \
  --finetune_epochs 3 \
  --alpha 0.7 --temperature 4.0 \
  --lr 0.01 \
  --track_energy \
  --output_dir outputs_pruned
/////////////////////////////////
INT8 Quantization 
" python scripts/quantize_student_fx_int8.py \  
--student_ckpt outputs_pruned_best/student_pruned_best.pt \  
--student resnet18 \
--img_size 128 \ 
--calib_batches 200 \
--batch_size 64 \  
--num_workers 2 \ 
--output_dir outputs_int8" 
