# 수정 사항
## utils/tools.py
  np.Inf → np.inf
## run.py
  args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False → comment out
