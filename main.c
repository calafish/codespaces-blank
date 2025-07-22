#include "audio_processor.h"
#include <getopt.h>
#include <math.h>

void print_usage(const char *program_name) {
    printf("使用方法: %s [选项] <音频文件>\n", program_name);
    printf("选项:\n");
    printf("  -r, --rate <采样率>    目标采样率 (默认: 44100)\n");
    printf("  -h, --help            显示此帮助信息\n");
    printf("  -i, --info            只显示文件信息，不处理数据\n");
    printf("\n");
    printf("支持的音频格式:\n");
    printf("  WAV, AIFF, AU, FLAC, OGG, MP3 等多种格式\n");
    printf("\n");
    printf("示例:\n");
    printf("  %s audio.wav                    # 读取并归一化\n", program_name);
    printf("  %s -r 48000 audio.wav           # 重采样到48kHz\n", program_name);
    printf("  %s -i audio.wav                 # 只显示文件信息\n", program_name);
}

int main(int argc, char *argv[]) {
    int target_samplerate = 44100;
    int info_only = 0;
    
    // 解析命令行参数
    static struct option long_options[] = {
        {"rate", required_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        {"info", no_argument, 0, 'i'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "r:hi", long_options, &option_index)) != -1) {
        switch (c) {
            case 'r':
                target_samplerate = atoi(optarg);
                if (target_samplerate <= 0) {
                    fprintf(stderr, "无效的采样率: %s\n", optarg);
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'i':
                info_only = 1;
                break;
            case '?':
                print_usage(argv[0]);
                return 1;
            default:
                break;
        }
    }
    
    // 检查是否提供了音频文件参数
    if (optind >= argc) {
        fprintf(stderr, "错误: 请提供音频文件路径\n");
        print_usage(argv[0]);
        return 1;
    }
    
    const char *filename = argv[optind];
    
    // 创建音频处理器
    AudioProcessor *processor = audio_processor_create();
    if (!processor) {
        fprintf(stderr, "创建音频处理器失败\n");
        return 1;
    }
    
    // 打开音频文件
    if (audio_processor_open_file(processor, filename) != 0) {
        audio_processor_destroy(processor);
        return 1;
    }
    
    // 打印文件信息
    audio_processor_print_info(processor);
    
    if (info_only) {
        printf("仅显示信息模式，程序结束\n");
        audio_processor_destroy(processor);
        return 0;
    }
    
    // 读取并归一化音频数据
    printf("\n正在读取和归一化音频数据...\n");
    AudioData *audio_data = audio_processor_read_and_normalize(processor);
    if (!audio_data) {
        fprintf(stderr, "读取音频数据失败\n");
        audio_processor_destroy(processor);
        return 1;
    }
    
    // 检查是否需要重采样
    AudioData *final_data = audio_data;
    if (audio_data->samplerate != target_samplerate) {
        printf("\n正在重采样到 %d Hz...\n", target_samplerate);
        final_data = audio_processor_resample(processor, audio_data, target_samplerate);
        if (!final_data) {
            fprintf(stderr, "重采样失败\n");
            audio_data_destroy(audio_data);
            audio_processor_destroy(processor);
            return 1;
        }
    }
    
    // 显示处理结果
    printf("\n=== 处理结果 ===\n");
    printf("最终音频数据:\n");
    printf("  帧数: %d\n", final_data->frames);
    printf("  声道数: %d\n", final_data->channels);
    printf("  采样率: %d Hz\n", final_data->samplerate);
    printf("  时长: %.2f 秒\n", final_data->duration);
    printf("  总样本数: %d\n", final_data->frames * final_data->channels);
    
    // 显示前几个样本值作为示例
    printf("\n前10个样本值:\n");
    int samples_to_show = (final_data->frames * final_data->channels > 10) ? 10 : 
                         (final_data->frames * final_data->channels);
    for (int i = 0; i < samples_to_show; i++) {
        printf("  样本[%d]: %.6f\n", i, final_data->data[i]);
    }
    
    // 计算音频统计信息
    float min_val = final_data->data[0], max_val = final_data->data[0];
    double sum = 0.0, sum_squares = 0.0;
    int total_samples = final_data->frames * final_data->channels;
    
    for (int i = 0; i < total_samples; i++) {
        float val = final_data->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        sum_squares += val * val;
    }
    
    double mean = sum / total_samples;
    double variance = (sum_squares / total_samples) - (mean * mean);
    double rms = sqrt(sum_squares / total_samples);
    
    printf("\n音频统计信息:\n");
    printf("  最小值: %.6f\n", min_val);
    printf("  最大值: %.6f\n", max_val);
    printf("  均值: %.6f\n", mean);
    printf("  RMS: %.6f\n", rms);
    printf("  方差: %.6f\n", variance);
    
    // 清理资源
    if (final_data != audio_data) {
        audio_data_destroy(audio_data);
    }
    audio_data_destroy(final_data);
    audio_processor_destroy(processor);
    
    printf("\n音频处理完成！\n");
    return 0;
}
