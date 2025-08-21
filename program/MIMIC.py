import kagglehub
    
    # Authenticate
    kagglehub.login() # This will prompt you for your credentials.
    # We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate
    
    # Download latest version
    path = kagglehub.model_download("google/gemma/pyTorch/2b")
    
    # Download specific version (here version 1)
    path = kagglehub.model_download("google/gemma/pyTorch/2b/1")
    
    print("Path to model files:", path)