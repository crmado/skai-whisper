import time
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

print("设备：", device)

# 载入模型和处理器
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)


# 预处理音频并获取模型输入
def preprocess_audio(audio_file_path):
    audio_input, sampling_rate = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs


# 转录音频文件
def transcribe_audio_file(audio_file_path):
    inputs = preprocess_audio(audio_file_path)
    input_features = inputs.input_features.to(device)

    # 記錄開始時間
    start_time = time.time()

    # 使用 generate 方法進行推理，獲取預測的標識符
    predicted_ids = model.generate(input_features)

    # 記錄結束時間並計算推理所花費的時間
    end_time = time.time()
    inference_time = end_time - start_time

    # 解碼預測的標識符以獲得轉錄文本
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription, inference_time


# 测试函数
if __name__ == "__main__":
    audio_file_path = "mac.wav"
    transcription = transcribe_audio_file(audio_file_path)
    print("转录结果：", transcription)
    print("推理时间單位：", transcription[1], "秒")
