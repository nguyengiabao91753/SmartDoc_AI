import httpx, os, time

# Thay bằng Key sk-or-v1-... của bạn
API_KEY = os.getenv("LLM_API_KEY", "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")
URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY.strip()}",
    "Content-Type": "application/json",
}

# Đảm bảo tên model khớp hoàn toàn với ảnh bạn chụp lúc nãy
data = {
    "model": "liquid/lfm-2.5-1.2b-instruct:free", 
    "messages": [
        {"role": "user", "content": "Hello, can you hear me?"}
    ]
}

print(f"Đang gửi yêu cầu tới model: {data['model']}...")

try:
    with httpx.Client() as client:
        # Gửi request với tham số json=data để httpx tự xử lý header và body
        response = client.post(URL, headers=headers, json=data, timeout=30.0)
        
        for i in range(60): # Thử lại tối đa 3 lần
            response = client.post(URL, headers=headers, json=data, timeout=30.0)
            if response.status_code == 200:
                print(response.json())
                print("Thành công!")
                break
            elif response.status_code == 429:
                print(f"Server bận, đang thử lại lần {i+1} sau 5 giây...")
                time.sleep(5) 
            else:
                print(f"Lỗi {response.status_code}")
                break
            
except Exception as e:
    print(f"Lỗi kết nối hệ thống: {e}")