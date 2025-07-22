# 音频处理器 (Audio Processor)

一个基于libsndfile和libsamplerate的音频处理工具，支持读取多种音频格式，进行PCM数据归一化处理，并支持采样率转换。

## 功能特性

- **多格式支持**: 支持WAV, AIFF, FLAC, OGG等多种音频格式
- **PCM数据提取**: 将音频数据读取为float数组格式
- **归一化处理**: 自动对音频数据进行归一化，防止削波
- **采样率转换**: 使用高质量算法进行采样率转换
- **详细信息显示**: 显示音频文件的详细格式信息和统计数据

## 依赖库

- `libsndfile`: 用于读取各种音频格式
- `libsamplerate`: 用于高质量采样率转换

## 安装依赖

### Ubuntu/Debian 系统
```bash
make install-deps
```

### CentOS/RHEL/Fedora 系统
```bash
make install-deps-rpm
```

### 手动安装
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1-dev libsamplerate0-dev

# CentOS/RHEL
sudo yum install libsndfile-devel libsamplerate-devel

# Fedora
sudo dnf install libsndfile-devel libsamplerate-devel
```

## 编译

```bash
make
```

编译调试版本：
```bash
make debug
```

## 使用方法

### 基本语法
```bash
./audio_processor [选项] <音频文件>
```

### 选项说明
- `-r, --rate <采样率>`: 指定目标采样率 (默认: 44100)
- `-i, --info`: 只显示文件信息，不处理数据
- `-h, --help`: 显示帮助信息

### 使用示例

1. **显示音频文件信息**：
```bash
./audio_processor -i audio.wav
```

2. **读取并归一化音频数据**：
```bash
./audio_processor audio.wav
```

3. **重采样到指定采样率**：
```bash
./audio_processor -r 48000 audio.wav
```

4. **处理FLAC文件并转换采样率**：
```bash
./audio_processor -r 22050 music.flac
```

## 输出信息

程序会显示以下信息：
- 音频文件的详细格式信息
- 处理过程中的状态信息
- 归一化处理的统计数据
- 重采样的转换信息
- 最终音频数据的统计信息（最小值、最大值、均值、RMS等）

## 支持的音频格式

- **无损格式**: WAV, AIFF, FLAC, AU
- **有损格式**: OGG Vorbis
- **其他格式**: CAF, W64, RF64等

具体支持的格式取决于系统安装的libsndfile版本。

## 代码结构

- `audio_processor.h`: 头文件，定义数据结构和函数接口
- `audio_processor.c`: 核心实现，包含音频读取、归一化和重采样功能
- `main.c`: 主程序，处理命令行参数和用户交互
- `Makefile`: 编译配置文件

## 核心数据结构

```c
// 音频数据结构
typedef struct {
    float *data;           // PCM数据数组
    int frames;            // 帧数
    int channels;          // 声道数
    int samplerate;        // 采样率
    double duration;       // 时长（秒）
} AudioData;

// 音频处理器结构
typedef struct {
    SNDFILE *file;         // libsndfile文件句柄
    SF_INFO sf_info;       // 文件信息
    SRC_STATE *src_state;  // libsamplerate状态
    int target_samplerate; // 目标采样率
} AudioProcessor;
```

## 清理

```bash
make clean
```

## 注意事项

1. 确保音频文件路径正确且文件可读
2. 大文件处理可能需要较长时间和较多内存
3. 重采样质量设置为最高质量，处理速度相对较慢但质量最佳
4. 程序会自动检测并处理各种音频格式的差异

## 错误处理

程序包含完整的错误处理机制：
- 文件打开失败
- 内存分配失败
- 音频格式不支持
- 重采样参数错误

所有错误都会显示详细的错误信息以便调试。
