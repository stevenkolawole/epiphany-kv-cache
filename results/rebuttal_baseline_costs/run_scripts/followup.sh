# Runs after grid.sh: reruns invalidated cells.
until grep -q GRID_DONE ~/rebuttal_baseline_costs/status.txt; do sleep 60; done
R=/opt/dlami/nvme/kvcache_cost/run.sh
$R rkv steady_rkv --mode steady --method rkv
echo FOLLOWUP_DONE >> ~/rebuttal_baseline_costs/status.txt
