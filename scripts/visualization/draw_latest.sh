python scripts/parse_chunggeng_tts.py
python scripts/parse_other_center_tts.py
python scripts/parse_seer_tts.py
# 画 km 曲线
# python scripts/visualization/apply_thresholds_km_plot.py \
#   --threshold-json scripts/ecdf_train_thresholds.json \
#   --csv-prefix df_ours \
#   --base-dir data/visualization \
#   --out-dir visualization_res \
#   --ecdf-path scripts/ecdf_train.pkl   # 若 JSON 为 cdf_thresholds 时必填

python scripts/visualization/apply_thresholds_km_plot.py \
  --threshold-json scripts/ecdf_train_thresholds.json \
  --csv-prefix df_ours \
  --base-dir data/visualization \
  --out-dir visualization_res \
  --ecdf-path scripts/ecdf_train.pkl  \
  --risk-interval 12

# 画 roc
python scripts/visualization/draw_roc_latest.py --base_dir data/visualization --colors "#4B3F72, #FFD373, #80C5A2, #C0392B, #468BCA, #D9C2DD, #5F5F5E, #008A45, #F27873, #00A9A5, #B384BA, #7DD2F6" #
# 画 cindex-by-task
python scripts/visualization/draw_cindex_latest.py --colors "#4B3F72, #FFD373, #80C5A2, #C0392B, #468BCA, #D9C2DD, #5F5F5E, #008A45, #F27873, #00A9A5, #B384BA, #7DD2F6" #
# python km_multi_schemes.py -i scripts/visualization/staging_survival_merged.jsonl -o scripts/visualization/visual_fin/km_ext_all_external --seed 2010
# 画 high/low risk line
python scripts/visualization/risk_line.py #
# python plot_internal_external_abcd_fast.py 
# 
# 画香港分期里那个 '假设 km' 曲线 （不同对比的方法）
# python scripts/visualization/km_multi_schemes_seed_dfs.py -i data/visualization/staging_survival_merged.jsonl -o visualization_res/km_ours_and_tranditional_13615 --seed 13615 --schemes OURS,BCLC,CNLC # old version
python scripts/visualization/km_multi.py -i data/visualization/staging_survival_merged.jsonl -o visualization_res/km_ours_and_tranditional_13615_new --schemes OURS,BCLC,CNLC --ecdf scripts/ecdf_train.pkl --ticks 12 --seed 13615 --axes lb
# python km_multi_schemes.py -i data.jsonl -o out/km_plot --schemes OURS,BCLC,CNLC --axes lb
# 画香港分期里那个 '假设 km' 曲线 （不同对比的方法）
# python scripts/visualization/km_multi_schemes_seed_dfs.py -i data/visualization/staging_survival_merged.jsonl -o visualization_res/km_all_13615 --seed 13615 #
python scripts/visualization/km_multi.py -i data/visualization/staging_survival_merged.jsonl -o visualization_res/km_ours_and_tranditional_13615_new --ecdf scripts/ecdf_train.pkl --ticks 12 --seed 13615
#   # --scan 12000:16000 \
#   # --workers 32 \
#   # --prefer median \
#   # --dominance soft --dom-weight 3.0 --dom-eps 0.0 --dom-grid 400 \
#   # --overall-weight 0.2
# 画不同中心的 cindex
# python cindex_each_center.py --compare-modes "p70_event,p60_event,fixed:24" --ipcw-gmin 0.05 --ipcw-wclip-quantile 0.95 --ipcw-normalize --n-boot 200 --colors "#008A45,#468BCA,#7DD2F6,#80C5A2,#00A9A5,#5F5F5E,#B384BA,#4B3F72,#D9C2DD,#F27873,#C0392B,#FFD373" #
# 画 scripts/visualization/visual_fin_latest/visual_finvisual_finours_internal_external_abcd.png（有点问题，需要 debug）
# python scripts/visualization/plot_internal_external_abcd_fast.py #
# 算表 3/4 不生成图
python scripts/visualization/make_stats_tables.py --base_dir data/visualization --internal_out visualization_res/internal_stats.csv --external_out visualization_res/external_stats.csv --n_boot 1000 --seed 20251012 #
# 画 topn 柱状图
python scripts/visualization/plot_topn_inclusion_metric.py \
  --cg3 data/visualization/staging_survival_cg_3.jsonl \
  --multicenter data/visualization/staging_survival_multicenter.jsonl \
  --seer data/visualization/staging_survival_seer_w_ml.jsonl
# 画 topn-threshold 折线图 - external
python scripts/visualization/plot_topn_external_only.py --colors "#4B3F72, #FFD373, #80C5A2, #C0392B, #468BCA, #D9C2DD, #5F5F5E, #008A45, #F27873, #00A9A5, #B384BA, #7DD2F6" #
# 画 topn-threshold 折线图 - internal
python scripts/visualization/draw_seer_different_threshold.py --colors "#4B3F72, #FFD373, #80C5A2, #C0392B, #468BCA, #D9C2DD, #5F5F5E, #008A45, #F27873, #00A9A5, #B384BA, #7DD2F6" #
python scripts/parse_chunggeng_tts_sft.py
python scripts/parse_other_center_tts_sft.py
python scripts/parse_seer_tts_sft.py

python scripts/visualization/eval_ours_vs_sft_cindex.py \
  --bar-colors "#8DA0CB,#FED88A,#90D7C6,#8CC0F2,#D7B8E6,#B0BEC5,#C5E1A5" \
  --pos-cap-colors "#4B5F91,#CC9E00,#2B8F6B,#468BCA,#9B5FB5,#5F6D75,#7CA355" \
  --neg-colors "#E39A6C,#4B5F91,#D9584A,#E67E22,#6BBF3E,#D9796E,#5B7AD9"
# 60 例对应的三张图
python scripts/visualization/cal_scoring_by_doctor.py scripts/visualization/topn_scores.csv scripts/visualization/reason_scores.csv visualization_res\
  --colors "#4B3F72,#468BCA,#00A9A5,#80C5A2,#00A9A5,#5F5F5E,#B384BA,#4B3F72,#D9C2DD,#F27873,#C0392B,#FFD373" 
# 60 例对应的有无 llm 辅助的图 <outdir>/combined_4panels.png 为4合一
python scripts/visualization/plot_from_per_case.py \
  --per-case scripts/visualization/per_case.csv \
  --outdir visualization_res \
  --time-filter iqr0.5
# 生成 sft 模型比较结果 <out_json>
python scripts/visualization/batch_eval_tts_nontts_gt_from_staging.py \
  --models qwen_8b r1_qwen_32b r1_qwen_7b qwq_32b qwen_32b \
  --out_json visualization_res/eval_summary_tts_vs_nontts.json \
  --workers 12
# 利用 sft 比较结果 <out_json> 画图
python scripts/visualization/compare_methods_summary.py \
  --inputs "visualization_res/eval_summary_tts_vs_nontts.json" \
  --internal "seer" \
  --outdir "visualization_res" \
  --model-colors "#4B3F72,#468BCA,#80C5A2,#FFD373,#5F5F5E,#B384BA,#00A9A5,#F27873" \
  --ymax-topn 1.2 \
  --ymax-cidx 0.90

python scripts/visualization/draw_scatter.py render \
  --auc-csv data/visualization/auc_scatter_points.csv \
  --acc-csv data/visualization/acc_scatter_points.csv \
  --meta data/visualization/scatter_meta.json \
  --outdir visualization_res \
  --model-time-sec 6 \
  --model-time-sec-acc 6