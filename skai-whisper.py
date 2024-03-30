import numpy as np
import sounddevice as sd
import torch
import queue
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa

# 初始化錄音隊列
q = queue.Queue()

# 選擇設備
if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

# 載入模型和處理器
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# 設定錄音參數
fs = 16000  # 採樣率


# 定義錄音的回調函數
def callback(indata, frames, time, status):
    q.put(indata.copy())


# 初始化錄音流
stream = sd.InputStream(callback=callback, samplerate=fs, channels=1)
print("準備好了，點擊按鈕開始錄音。")


# 開始錄音的函數
def start_recording():
    stream.start()
    print("開始錄音...")


# 停止錄音並進行轉寫的函數
def stop_and_transcribe():
    stream.stop()
    print("錄音結束。")

    # 從隊列中獲取所有錄音數據並合併
    recording = np.concatenate(list(q.queue))
    q.queue.clear()

    # 使用 librosa 預處理音頻
    audio_input, _ = librosa.effects.trim(recording.astype(np.float32), top_db=30)
    input_values = processor(audio_input, sampling_rate=fs, return_tensors="pt").input_values.to(device)

    # 進行模型推理
    with torch.no_grad():
        logits = model(input_values).logits

    # 解碼獲得預測文字
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    print("轉錄結果：", transcription)


if __name__ == "__main__":
    try:
        while True:
            input("按 Enter 開始錄音...")
            start_recording()
            input("按 Enter 停止錄音...")
            stop_and_transcribe()
    except KeyboardInterrupt:
        print("程式結束。")
        stream.close()
