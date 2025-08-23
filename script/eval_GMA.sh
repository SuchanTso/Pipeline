MODEL="model/GraphMAE_re.pt"
FigPath="GraphMAE_re"
# python src/train/eval_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/tt.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/tt.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net1.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net1.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net2.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net2.log
python src/train/eval_GMA.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/EPANET/example-networks/Net3.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net3_re.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Net3_(BWSN-2)_Morph_Error_Free_1s-WQ.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net3_Morph_Error_Free_1s.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Net3_EPANET-EXAMPLE_No_Demand_Change.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Net3_EPANET.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/Micropolis_TEVA-SPOT_Adjusted_PumpCurve3&4.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Micropolis_TEVA.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond_skeleton.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Richmond_skeleton.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/Richmond.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/Richmond.log
# python src/train/eval.py -m $MODEL -d "/data/zsc/Pipeline/data/epaNet/data-water-epanet/VanZyl.inp" -p $FigPath -e 100 -l 10 --hours_analysis 72 > results/VanZyl.log