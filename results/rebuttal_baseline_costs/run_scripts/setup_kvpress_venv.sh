set -e
cd /opt/dlami/nvme/kvcache_cost
python3 -m venv venv_kvpress
V="venv_kvpress/bin/pip"
$V install -q --upgrade pip
$V install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
ABI=$(venv_kvpress/bin/python -c "import torch;print(\"TRUE\" if torch._C._GLIBCXX_USE_CXX11_ABI else \"FALSE\")")
$V install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl"
$V install -q "transformers==4.57.1" "kvpress==0.5.5" accelerate datasets sympy latex2sympy2_extended
venv_kvpress/bin/python -c "import torch,transformers,flash_attn,kvpress;print(\"STACK\",torch.__version__,transformers.__version__,flash_attn.__version__,kvpress.__version__ if hasattr(kvpress,\"__version__\") else \"?\")"
echo SETUP_DONE
