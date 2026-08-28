try:
    import torch
    print(f"✅ PyTorch 加载成功！版本：{torch.__version__}")
    print(f"✅ CUDA 状态：{torch.cuda.is_available()}（CPU版显示False正常）")
    print(f"✅ CUDA 设备数：{torch.cuda.device_count()}")
    print(f"✅ CUDA 设备名称：{torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch 加载失败：{e}")