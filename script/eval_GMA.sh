MODEL="model/temporal_GMAEv5.pt"
FigPath="temporal_GMAEv5"
# python src/train/eval_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/tt.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/tt.log
# python src/train/eval_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net1.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net1_temporal_GMAEv5.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net2.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net2.log
# python src/train/eval_GMA.py -m "model/EST_MAE_ANYTOWN.pt" -d "/data/zsc/Pipeline/data/epaNet/Anytown.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/EST_MAE_Anytown_0.5.log
# python src/train/eval_GMA.py -m "model/EST_MAE_LTOWN.pt" -d "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/EST_MAE_LTOWN_0.5.log
# python src/train/eval_GMA.py -m "model/EST_MAE_CTOWN.pt" -d "/data/zsc/Pipeline/data/epaNet/CTOWN.INP" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/EST_MAE_CTOWN_0.5.log

# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Net3_(BWSN-2)_Morph_Error_Free_1s-WQ.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net3_Morph_Error_Free_1s.log
# python src/train/eval_GMA.py -m "model/EST_MAE_NET3.pt" -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net3.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net3_EPANET_0.98.log
python src/train/eval_GMA.py -m "model/EST_MAE_EXN.pt" -d "/data/zsc/Pipeline/data/epaNet/L-TOWN.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/EXN_4_l_town_0.98.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Micropolis_TEVA-SPOT_Adjusted_PumpCurve3&4.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Micropolis_TEVA.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond_skeleton.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Richmond_skeleton.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Richmond.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/VanZyl.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/VanZyl.log