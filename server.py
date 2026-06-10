import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai
import uuid

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
    api_key: str
    theme: str = ""

m4a_folder = "Project/m4aFolder/"
os.makedirs(m4a_folder, exist_ok=True)

# 🌟 루트 접속 시 index.html 렌더링
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/get_subtitles")
def get_subtitles(req: SubtitleRequest): 
    print(f"\n{req.url} 오디오 다운로드 중...")

    temp_filename = f"audio_{uuid.uuid4().hex}.m4a"
    current_file_path = os.path.join(m4a_folder, temp_filename)
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': current_file_path,
        'quiet': True,
        'noplaylist': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            ydl.extract_info(req.url, download=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"다운로드 중 에러 발생 : {e}")

    try:
        print("Gemini에게 파일 보내는 중...")
        genai.configure(api_key=req.api_key)

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
            - 프레임이나 밀리초 단위는 표시하지 마.
            - 예시 : [00:30] 영상번역내용 / [13:10] 영상번역내용.

        2. 평문(Plain Text)만 사용 :
            - **강조** 처리를 포함한 모든 마크다운(Markdown) 문법을 절대 사용하지 마.
            - 제목(#), 리스트(-), 볼드체(**) 없이 오직 순수한 텍스트로만 출력해.

        3. 내용 : 중요한 코드 개념을 포함하여 상세하게 설명하되, 위 두 가지 규칙을 엄격하게 지켜줘.
        """

        final_prompt = f"{intro_prompt}\n\n{rules_prompt}"

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([audio_file, final_prompt])

        if current_file_path and os.path.exists(current_file_path):
            os.remove(current_file_path)
            print("임시 오디오 파일 삭제 완료.")

        try:
            genai.delete_file(audio_file.name)
        except:
            pass

        print("자막 생성 완료!")
        return {"subtitle_text": response.text}
    
    except Exception as e:
        if current_file_path and os.path.exists(current_file_path):
            os.remove(current_file_path)
        raise HTTPException(status_code=500, detail=f"Gemini 처리 중 에러 발생 : {e}")