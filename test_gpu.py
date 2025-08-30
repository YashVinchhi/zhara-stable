#!/usr/bin/env python3
"""
GPU Detection Test Script
This script tests if NVIDIA GPU is properly detected and available for PyTorch
"""

import torch
import sys

def test_gpu_detection():
    """Test comprehensive GPU detection"""
    print("🔍 Testing GPU Detection...")
    print("=" * 50)

    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

        # Device count
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")

        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")

        # Device properties
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n🎮 GPU {i}: {props.name}")
            print(f"   Memory: {props.total_memory / (1024**3):.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multiprocessors: {props.multi_processor_count}")

            # Memory info
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)
                print(f"   Memory Allocated: {memory_allocated:.1f} MB")
                print(f"   Memory Reserved: {memory_reserved:.1f} MB")
            except Exception as e:
                print(f"   Memory info error: {e}")

        # Test GPU computation
        print("\n🧪 Testing GPU Computation...")
        try:
            # Create a tensor on GPU
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')

            # Perform computation
            z = torch.matmul(x, y)

            print("✅ GPU computation test PASSED")
            print(f"   Result tensor shape: {z.shape}")
            print(f"   Result device: {z.device}")

        except Exception as e:
            print(f"❌ GPU computation test FAILED: {e}")

        # Driver version (if available)
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                driver_info = result.stdout.strip().split(', ')
                if len(driver_info) >= 2:
                    print(f"\n📋 NVIDIA Driver Version: {driver_info[0]}")
                    print(f"📋 GPU Name from nvidia-smi: {driver_info[1]}")
        except Exception as e:
            print(f"\n⚠️  Could not get nvidia-smi info: {e}")

    else:
        print("❌ CUDA is not available")
        print("Possible reasons:")
        print("- NVIDIA drivers not installed")
        print("- PyTorch not compiled with CUDA support")
        print("- CUDA runtime not installed")

    print("\n" + "=" * 50)
    return cuda_available

if __name__ == "__main__":
    test_gpu_detection()
