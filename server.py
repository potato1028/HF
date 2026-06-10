import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import uuid
import requests 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubtitleRequest(BaseModel):
    url: str
    api_key: str = ""
    theme: str = ""

m4a_folder = "Project/m4aFolder/"
os.makedirs(m4a_folder, exist_ok=True)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/get_subtitles")
def get_subtitles(req: SubtitleRequest): 
    public_api_key = os.getenv("PUBLIC_GEMINI_API_KEY")
    use_public_key = False
    
    if not req.api_key.strip():
        if not public_api_key:
            raise HTTPException(status_code=400, detail="서버에 공용 API 키가 설정되지 않았습니다. 개인 API 키를 입력해주세요.")
        actual_api_key = public_api_key
        use_public_key = True
    else:
        actual_api_key = req.api_key.strip()

    print(f"\n{req.url} 오디오 다운로드 중 (Cobalt API 사용)...")

    temp_filename = f"audio_{uuid.uuid4().hex}.mp3"
    current_file_path = os.path.join(m4a_folder, temp_filename)

    # 🌟 Cobalt API를 사용하여 오디오 추출
    cobalt_api_url = "https://api.cobalt.tools/api/json"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "AI-Youtube-Subtitle-App" # 봇 차단 방지를 위한 User-Agent
    }
    payload = {
        "url": req.url,
        "isAudioOnly": True,
        "aFormat": "mp3"
    }

    try:
        # 1. Cobalt API에 다운로드 링크 요청
        response = requests.post(cobalt_api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "stream" or data.get("status") == "redirect":
                audio_download_link = data.get("url")
                
                # 2. 반환받은 링크에서 실제 MP3 다운로드
                audio_response = requests.get(audio_download_link)
                with open(current_file_path, "wb") as f:
                    f.write(audio_response.content)
                print("Cobalt API 오디오 다운로드 완료.")
            else:
                raise Exception("Cobalt API에서 유효한 오디오 링크를 받지 못했습니다.")
        else:
            raise Exception(f"Cobalt 서버 에러: {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오디오 추출 중 에러 발생 : {e}")

    # --- 이하 Gemini AI 처리 로직 ---
    try:
        print("Gemini에게 파일 보내는 중...")
        genai.configure(api_key=actual_api_key)

        audio_file = genai.upload_file(path=current_file_path)

        while audio_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(5)
            audio_file = genai.get_file(audio_file.name)
        
        if audio_file.state.name != "ACTIVE":
            raise Exception(f"파일 처리 실패 : {audio_file.state.name}")
    
        print("\n번역 및 요약 시작...")

        if req.theme.strip():
            intro_prompt = f"이 영상은 {req.theme}에 대한 설명이야. 내용을 한국어로 번역해줘"
        else:
            intro_prompt = "이 영상의 내용을 한국어로 번역해서 설명해줘"

        rules_prompt = """
        [반드시 지켜야 할 출력 규칙]
        1. 시간 표시 필수 : 문장 앞에 영상의 위치를 반드시 [MM:SS] (분:초) 형식으로 적어줘.
            - 주의 : '초'단위는 절대 60을 넘을 수 없어.
            - 예시 : [00:30] 영상번역내용 / [13:10] 영상번역내용.

        2. 평문(Plain Text)만 사용 :
            - **강조** 처리를 포함한 모든 마크다운(Markdown) 문법을 절대 사용하지 마.

        3. 내용 : 중요한 코드 개념을 포함하여 상세하게 설명하되, 위 두 가지 규칙을 엄격하게 지켜줘.
        """

        final_prompt = f"{intro_prompt}\n\n{rules_prompt}"

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([audio_file, final_prompt])

    except Exception as e:
        if current_file_path and os.path.exists(current_file_path):
            os.remove(current_file_path)
            
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg or "exhausted" in error_msg:
            if use_public_key:
                raise HTTPException(status_code=429, detail="PUBLIC_API_EXHAUSTED")
            else:
                raise HTTPException(status_code=429, detail="입력하신 개인 API 키의 한도가 초과되었습니다.")
                
        raise HTTPException(status_code=500, detail=f"Gemini 처리 중 에러 발생 : {e}")

    if current_file_path and os.path.exists(current_file_path):
        os.remove(current_file_path)
        print("임시 오디오 파일 삭제 완료.")

    try:
        genai.delete_file(audio_file.name)
    except:
        pass

    print("자막 생성 완료!")
    return {"subtitle_text": response.text}