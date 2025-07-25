#!/usr/bin/env python3
"""
验证 Fish Speech 与 vLLM 集成的脚本
"""

import sys
import os
sys.path.insert(0, '/workspace/vllm')

def test_imports():
    """测试模块导入"""
    print("Testing imports...")
    
    try:
        from vllm.model_executor.models.fish_speech import (
            FishSpeechConfig,
            FishSpeechForCausalLM,
            FishSpeechAttention,
            FishSpeechMLP,
            FishSpeechDecoderLayer,
            FishSpeechModel,
        )
        print("✅ Fish Speech modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """测试配置类"""
    print("\nTesting configuration...")
    
    try:
        from vllm.model_executor.models.fish_speech import FishSpeechConfig
        
        # 测试默认配置
        config = FishSpeechConfig()
        print(f"✅ Default config created: model_type={config.model_type}")
        
        # 测试自定义配置
        custom_config = FishSpeechConfig(
            vocab_size=50000,
            n_layer=24,
            n_head=16,
            dim=2048,
        )
        print(f"✅ Custom config created: vocab_size={custom_config.vocab_size}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_registration():
    """测试模型注册"""
    print("\nTesting model registration...")
    
    try:
        from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
        
        if "FishSpeechForCausalLM" in _TEXT_GENERATION_MODELS:
            module_name, class_name = _TEXT_GENERATION_MODELS["FishSpeechForCausalLM"]
            print(f"✅ Fish Speech model registered: {module_name}.{class_name}")
            return True
        else:
            print("❌ Fish Speech model not found in registry")
            return False
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_converter_tool():
    """测试转换工具"""
    print("\nTesting converter tool...")
    
    converter_path = "/workspace/vllm/tools/fish_speech_converter.py"
    if os.path.exists(converter_path):
        print(f"✅ Converter tool exists: {converter_path}")
        return True
    else:
        print(f"❌ Converter tool not found: {converter_path}")
        return False

def test_example_script():
    """测试示例脚本"""
    print("\nTesting example script...")
    
    example_path = "/workspace/examples/fish_speech_inference.py"
    if os.path.exists(example_path):
        print(f"✅ Example script exists: {example_path}")
        return True
    else:
        print(f"❌ Example script not found: {example_path}")
        return False

def main():
    """主函数"""
    print("Fish Speech + vLLM 集成验证")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_model_registration,
        test_converter_tool,
        test_example_script,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！Fish Speech 与 vLLM 集成成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查实现")
        return 1

if __name__ == "__main__":
    sys.exit(main())