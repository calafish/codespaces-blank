#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <sndfile.h>
#include <samplerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// 函数声明
AudioProcessor* audio_processor_create(void);
void audio_processor_destroy(AudioProcessor *processor);
int audio_processor_open_file(AudioProcessor *processor, const char *filename);
void audio_processor_close_file(AudioProcessor *processor);
AudioData* audio_processor_read_and_normalize(AudioProcessor *processor);
AudioData* audio_processor_resample(AudioProcessor *processor, AudioData *input, int target_samplerate);
void audio_data_destroy(AudioData *audio_data);
void audio_processor_print_info(AudioProcessor *processor);

#endif // AUDIO_PROCESSOR_H
