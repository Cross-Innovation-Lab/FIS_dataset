# 前台训练任务一
bash scripts/run_fisnet.sh train task1

# 后台训练任务二（nohup）
bash scripts/run_fisnet.sh train task2 --bg

# 评估已有 checkpoint
bash scripts/run_fisnet.sh eval task1 checkpoints/fis_net_xxx/best.pt

# 直接传自定义配置
bash scripts/run_fisnet.sh train /path/to/custom.json

# 基线模型：BiLSTM+Attention / TimeSformer（任务一 & 二）
# python -m experiment.run train --config experiment/configs/bilstm_attn_task1.json
# python -m experiment.run train --config experiment/configs/bilstm_attn_task2.json
# python -m experiment.run train --config experiment/configs/timesformer_task1.json
# python -m experiment.run train --config experiment/configs/timesformer_task2.json