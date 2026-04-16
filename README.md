# 수정 사항(2026.04.16.)
## utils/tools.py
  np.Inf → np.inf
## run.py
  args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False → comment out
## experiments/exp_long_term_forecasting.py
  "Remove that saves the test results as a PDF every 20 steps."
