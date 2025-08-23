MODEL="model/GraphMAE_re.pt"
# python src/train/train_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/tt.inp" -e 100 -l 10 --hours_analysis 72
# python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net1.inp" -e 10 -l 10 --hours_analysis 72
# python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Net3_EPANET-EXAMPLE_No_Demand_Change.inp" -e 10 -l 10 --hours_analysis 72
# python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Net3_(BWSN-2)_Morph_Error_Free_1s-WQ.inp" -e 10 -l 10 --hours_analysis 72
python src/train/train_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net3.inp" -e 200 -l 10 --hours_analysis 72
## python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/VanZyl.inp" -e 100 -l 10 --hours_analysis 72
## python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Generated.inp" -e 100 -l 10 --hours_analysis 72
## python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond_skeleton.inp" -e 100 -l 10 --hours_analysis 72
## python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond.inp" -e 100 -l 10 --hours_analysis 72
## python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Micropolis_TEVA-SPOT_Adjusted_PumpCurve3&4.inp" -e 100 -l 10 --hours_analysis 72
# python src/train/train.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net2.inp" -e 100 -l 10 --hours_analysis 72