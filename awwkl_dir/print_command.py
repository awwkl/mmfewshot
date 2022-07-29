import os

# === Adjust these variables ===
config_name = 'attention-rpn_r50_c4_4xb2_bird_10shot-fine-tuning'
iter_num = 140
# ==============================

dataset = 'manhole' if 'manhole' in config_name else 'bird'
config_file = os.path.join('awwkl_dir/configs/', dataset, config_name + '.py')
work_dir = os.path.join('work_dirs/', config_name)
weights_file = os.path.join(work_dir, f'iter_{iter_num}.pth')
if iter_num == 0:
    weights_file = 'work_dirs/downloaded/attention-rpn_r50_c4_4xb2_coco_base-training_20211102_003348-da28cdfd.pth'
show_dir = os.path.join(work_dir, f'iter_{iter_num}' + '_nms')
out_path = os.path.join(show_dir, "result.pkl")
log_path = os.path.join(show_dir, 'eval.log')

if not os.path.exists(show_dir):
    os.makedirs(show_dir)

train_cmd = []
train_cmd.append('python tools/detection/train.py')
train_cmd.append(config_file)
# print('---')
# print(' '.join(train_cmd))

print('---')
test_cmd = []
test_cmd.append('python')
test_cmd.append('tools/detection/test.py')
test_cmd.append(config_file)
test_cmd.append(weights_file)
test_cmd.append('--out')
test_cmd.append(out_path)
test_cmd.append('--eval bbox')
test_cmd.append('--show --show-dir')
test_cmd.append(show_dir)
test_cmd.append(">")
test_cmd.append(log_path)
print(' '.join(test_cmd))
