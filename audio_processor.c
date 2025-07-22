#include "audio_processor.h"
#include <math.h>

// 创建音频处理器
AudioProcessor* audio_processor_create(void) {
    AudioProcessor *processor = malloc(sizeof(AudioProcessor));
    if (!processor) {
        fprintf(stderr, "内存分配失败\n");
        return NULL;
    }
    
    memset(&processor->sf_info, 0, sizeof(SF_INFO));
    processor->file = NULL;
    processor->src_state = NULL;
    processor->target_samplerate = 44100; // 默认采样率
    
    return processor;
}

// 销毁音频处理器
void audio_processor_destroy(AudioProcessor *processor) {
    if (!processor) return;
    
    audio_processor_close_file(processor);
    free(processor);
}

// 打开音频文件
int audio_processor_open_file(AudioProcessor *processor, const char *filename) {
    if (!processor || !filename) {
        fprintf(stderr, "无效的参数\n");
        return -1;
    }
    
    // 关闭之前打开的文件
    audio_processor_close_file(processor);
    
    // 打开新文件
    processor->file = sf_open(filename, SFM_READ, &processor->sf_info);
    if (!processor->file) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        fprintf(stderr, "libsndfile错误: %s\n", sf_strerror(NULL));
        return -1;
    }
    
    printf("成功打开文件: %s\n", filename);
    return 0;
}

// 关闭音频文件
void audio_processor_close_file(AudioProcessor *processor) {
    if (!processor) return;
    
    if (processor->file) {
        sf_close(processor->file);
        processor->file = NULL;
    }
    
    if (processor->src_state) {
        src_delete(processor->src_state);
        processor->src_state = NULL;
    }
}

// 读取音频数据并进行归一化
AudioData* audio_processor_read_and_normalize(AudioProcessor *processor) {
    if (!processor || !processor->file) {
        fprintf(stderr, "无效的处理器或文件未打开\n");
        return NULL;
    }
    
    AudioData *audio_data = malloc(sizeof(AudioData));
    if (!audio_data) {
        fprintf(stderr, "内存分配失败\n");
        return NULL;
    }
    
    // 设置音频数据信息
    audio_data->frames = processor->sf_info.frames;
    audio_data->channels = processor->sf_info.channels;
    audio_data->samplerate = processor->sf_info.samplerate;
    audio_data->duration = (double)audio_data->frames / audio_data->samplerate;
    
    // 分配内存用于存储PCM数据
    int total_samples = audio_data->frames * audio_data->channels;
    audio_data->data = malloc(total_samples * sizeof(float));
    if (!audio_data->data) {
        fprintf(stderr, "PCM数据内存分配失败\n");
        free(audio_data);
        return NULL;
    }
    
    // 读取音频数据
    sf_count_t read_count = sf_readf_float(processor->file, audio_data->data, audio_data->frames);
    if (read_count != audio_data->frames) {
        fprintf(stderr, "读取音频数据不完整: 期望 %d 帧，实际读取 %ld 帧\n", 
                audio_data->frames, read_count);
    }
    
    // 归一化处理
    float max_value = 0.0f;
    for (int i = 0; i < total_samples; i++) {
        float abs_value = fabsf(audio_data->data[i]);
        if (abs_value > max_value) {
            max_value = abs_value;
        }
    }
    
    // 如果最大值大于0，进行归一化
    if (max_value > 0.0f) {
        float scale = 1.0f / max_value;
        for (int i = 0; i < total_samples; i++) {
            audio_data->data[i] *= scale;
        }
        printf("音频数据已归一化，最大值: %.6f，缩放因子: %.6f\n", max_value, scale);
    } else {
        printf("音频数据为静音，无需归一化\n");
    }
    
    printf("成功读取音频数据: %d 帧, %d 声道, %d Hz\n", 
           audio_data->frames, audio_data->channels, audio_data->samplerate);
    
    return audio_data;
}

// 重采样音频数据
AudioData* audio_processor_resample(AudioProcessor *processor, AudioData *input, int target_samplerate) {
    if (!processor || !input || target_samplerate <= 0) {
        fprintf(stderr, "无效的重采样参数\n");
        return NULL;
    }
    
    // 如果采样率相同，直接返回复制的数据
    if (input->samplerate == target_samplerate) {
        printf("采样率相同，无需重采样\n");
        AudioData *output = malloc(sizeof(AudioData));
        if (!output) return NULL;
        
        *output = *input;
        int total_samples = input->frames * input->channels;
        output->data = malloc(total_samples * sizeof(float));
        if (!output->data) {
            free(output);
            return NULL;
        }
        memcpy(output->data, input->data, total_samples * sizeof(float));
        return output;
    }
    
    // 创建重采样器状态
    int error;
    SRC_STATE *src_state = src_new(SRC_SINC_BEST_QUALITY, input->channels, &error);
    if (!src_state) {
        fprintf(stderr, "创建重采样器失败: %s\n", src_strerror(error));
        return NULL;
    }
    
    // 计算重采样比率和输出帧数
    double ratio = (double)target_samplerate / input->samplerate;
    int output_frames = (int)(input->frames * ratio) + 1;
    
    // 分配输出数据内存
    AudioData *output = malloc(sizeof(AudioData));
    if (!output) {
        src_delete(src_state);
        return NULL;
    }
    
    output->channels = input->channels;
    output->samplerate = target_samplerate;
    output->frames = output_frames;
    output->duration = (double)output_frames / target_samplerate;
    
    int total_output_samples = output_frames * output->channels;
    output->data = malloc(total_output_samples * sizeof(float));
    if (!output->data) {
        free(output);
        src_delete(src_state);
        return NULL;
    }
    
    // 设置重采样数据
    SRC_DATA src_data;
    src_data.data_in = input->data;
    src_data.input_frames = input->frames;
    src_data.data_out = output->data;
    src_data.output_frames = output_frames;
    src_data.src_ratio = ratio;
    src_data.end_of_input = SF_TRUE;
    
    // 执行重采样
    error = src_process(src_state, &src_data);
    if (error != 0) {
        fprintf(stderr, "重采样失败: %s\n", src_strerror(error));
        audio_data_destroy(output);
        src_delete(src_state);
        return NULL;
    }
    
    // 更新实际的输出帧数
    output->frames = src_data.output_frames_gen;
    output->duration = (double)output->frames / target_samplerate;
    
    src_delete(src_state);
    
    printf("重采样完成: %d Hz -> %d Hz, %d 帧 -> %d 帧\n", 
           input->samplerate, target_samplerate, input->frames, output->frames);
    
    return output;
}

// 销毁音频数据
void audio_data_destroy(AudioData *audio_data) {
    if (!audio_data) return;
    
    if (audio_data->data) {
        free(audio_data->data);
    }
    free(audio_data);
}

// 打印音频信息
void audio_processor_print_info(AudioProcessor *processor) {
    if (!processor || !processor->file) {
        printf("无音频文件信息\n");
        return;
    }
    
    SF_INFO *info = &processor->sf_info;
    
    printf("=== 音频文件信息 ===\n");
    printf("帧数: %d\n", (int)info->frames);
    printf("采样率: %d Hz\n", info->samplerate);
    printf("声道数: %d\n", info->channels);
    printf("时长: %.2f 秒\n", (double)info->frames / info->samplerate);
    
    // 格式信息
    printf("格式: ");
    switch (info->format & SF_FORMAT_TYPEMASK) {
        case SF_FORMAT_WAV: printf("WAV"); break;
        case SF_FORMAT_AIFF: printf("AIFF"); break;
        case SF_FORMAT_AU: printf("AU"); break;
        case SF_FORMAT_RAW: printf("RAW"); break;
        case SF_FORMAT_PAF: printf("PAF"); break;
        case SF_FORMAT_SVX: printf("SVX"); break;
        case SF_FORMAT_NIST: printf("NIST"); break;
        case SF_FORMAT_VOC: printf("VOC"); break;
        case SF_FORMAT_IRCAM: printf("IRCAM"); break;
        case SF_FORMAT_W64: printf("W64"); break;
        case SF_FORMAT_MAT4: printf("MAT4"); break;
        case SF_FORMAT_MAT5: printf("MAT5"); break;
        case SF_FORMAT_PVF: printf("PVF"); break;
        case SF_FORMAT_XI: printf("XI"); break;
        case SF_FORMAT_HTK: printf("HTK"); break;
        case SF_FORMAT_SDS: printf("SDS"); break;
        case SF_FORMAT_AVR: printf("AVR"); break;
        case SF_FORMAT_WAVEX: printf("WAVEX"); break;
        case SF_FORMAT_SD2: printf("SD2"); break;
        case SF_FORMAT_FLAC: printf("FLAC"); break;
        case SF_FORMAT_CAF: printf("CAF"); break;
        case SF_FORMAT_WVE: printf("WVE"); break;
        case SF_FORMAT_OGG: printf("OGG"); break;
        case SF_FORMAT_MPC2K: printf("MPC2K"); break;
        case SF_FORMAT_RF64: printf("RF64"); break;
        default: printf("未知格式 (0x%x)", info->format & SF_FORMAT_TYPEMASK);
    }
    
    printf(" | ");
    switch (info->format & SF_FORMAT_SUBMASK) {
        case SF_FORMAT_PCM_S8: printf("8-bit PCM"); break;
        case SF_FORMAT_PCM_16: printf("16-bit PCM"); break;
        case SF_FORMAT_PCM_24: printf("24-bit PCM"); break;
        case SF_FORMAT_PCM_32: printf("32-bit PCM"); break;
        case SF_FORMAT_PCM_U8: printf("8-bit unsigned PCM"); break;
        case SF_FORMAT_FLOAT: printf("32-bit float"); break;
        case SF_FORMAT_DOUBLE: printf("64-bit double"); break;
        case SF_FORMAT_ULAW: printf("U-Law"); break;
        case SF_FORMAT_ALAW: printf("A-Law"); break;
        case SF_FORMAT_IMA_ADPCM: printf("IMA ADPCM"); break;
        case SF_FORMAT_MS_ADPCM: printf("Microsoft ADPCM"); break;
        case SF_FORMAT_GSM610: printf("GSM 6.10"); break;
        case SF_FORMAT_VOX_ADPCM: printf("Oki Dialogic ADPCM"); break;
        case SF_FORMAT_G721_32: printf("32kbs G721 ADPCM"); break;
        case SF_FORMAT_G723_24: printf("24kbs G723 ADPCM"); break;
        case SF_FORMAT_G723_40: printf("40kbs G723 ADPCM"); break;
        case SF_FORMAT_DWVW_12: printf("12 bit Delta Width Variable Word"); break;
        case SF_FORMAT_DWVW_16: printf("16 bit Delta Width Variable Word"); break;
        case SF_FORMAT_DWVW_24: printf("24 bit Delta Width Variable Word"); break;
        case SF_FORMAT_DWVW_N: printf("N bit Delta Width Variable Word"); break;
        case SF_FORMAT_DPCM_8: printf("8 bit differential PCM"); break;
        case SF_FORMAT_DPCM_16: printf("16 bit differential PCM"); break;
        case SF_FORMAT_VORBIS: printf("Xiph Vorbis"); break;
        default: printf("未知子格式 (0x%x)", info->format & SF_FORMAT_SUBMASK);
    }
    printf("\n");
    
    printf("可搜索: %s\n", info->seekable ? "是" : "否");
    printf("==================\n");
}
