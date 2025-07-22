CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LIBS = -lsndfile -lsamplerate -lm
TARGET = audio_processor
SOURCES = main.c audio_processor.c
OBJECTS = $(SOURCES:.c=.o)

# 默认目标
all: $(TARGET)

# 链接目标文件生成可执行文件
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LIBS)

# 编译源文件为目标文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理编译生成的文件
clean:
	rm -f $(OBJECTS) $(TARGET)

# 安装依赖 (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libsndfile1-dev libsamplerate0-dev

# 安装依赖 (CentOS/RHEL/Fedora)
install-deps-rpm:
	sudo yum install -y libsndfile-devel libsamplerate-devel

# 运行测试 (需要有测试音频文件)
test: $(TARGET)
	@echo "请提供音频文件进行测试，例如:"
	@echo "./$(TARGET) -h"
	@echo "./$(TARGET) -i your_audio_file.wav"
	@echo "./$(TARGET) -r 48000 your_audio_file.wav"

# 调试版本
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

.PHONY: all clean install-deps install-deps-rpm test debug
