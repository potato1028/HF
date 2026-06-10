import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

# 🌟 Gemini SDK 임포트
from google import genai
from google.genai import errors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    url: str
    api_key: str = ""
    tone: str = ""

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/translate_news")
def translate_news(req: NewsRequest): 
    # 1. API 키 결정 로직 (사용자 키 vs 공용 키)
    server_public_gemini_key = os.getenv("GEMINI_API_KEY")
    is_using_public_key = False
    
    if req.api_key.strip():
        final_api_key = req.api_key.strip()
    else:
        if not server_public_gemini_key:
            raise HTTPException(status_code=400, detail="서버 공용 토큰이 설정되지 않았습니다. 개인 Gemini API 키를 입력해 주세요.")
        final_api_key = server_public_gemini_key
        is_using_public_key = True

    print(f"\n{req.url} 뉴스 기사 크롤링 시작...")

    # 2. 뉴스 웹페이지 본문 크롤링
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(req.url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        
        if not article_text:
            raise Exception("기사 본문을 추출할 수 없습니다. 다른 링크로 시도해 주세요.")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"뉴스 기사 가져오기 실패 : {e}")

    # 3. Gemini API 호출
    try:
        print("Gemini 모델 분석 요청 중...")
        client = genai.Client(api_key=final_api_key)

        tone_instruction = f"번역 어조/타겟: {req.tone}" if req.tone.strip() else "번역 어조: 신뢰감 있는 전문 뉴스 앵커 스타일"

        system_instruction = f"""
        너는 20년 경력의 수석 해외 뉴스 전문 번역가이자 편집장이야.
        원문의 의미를 정확하게 파악하고, 한국어 원어민이 읽었을 때 전혀 어색함이 없는 매끄러운 기사를 작성해.

        [번역 품질 가이드라인]
        1. 직역을 엄격히 금지하며, 문맥과 뉘앙스를 살려 한국어 표현에 맞는 자연스러운 문장 구조(의역)로 재작성해.
        2. 관용구나 정치/사회적 표현은 기계적으로 바꾸지 말고, 한국 언론에서 쓰는 저널리즘 어투로 다듬어.
        3. 사람 이름, 회사명 등 중요한 고유명사는 첫 등장 시 한글 표기와 함께 괄호 안에 영문을 병기해. (예: 일론 머스크(Elon Musk))
        4. 문장은 간결하고 명확하게 끝맺음을 처리해.
        5. {tone_instruction}

        [출력 규칙]
        반드시 아래 두 가지 섹션으로만 나누어 한국어로 출력해. 각 섹션 제목은 굵은 글씨(**)로 표시해.
        
        **📌 핵심 3줄 요약**
        (기사의 가장 핵심적인 인사이트를 3개의 불릿 포인트로 간결하게 요약)

        **📰 전체 한글 번역**
        (위 번역 가이드라인을 철저히 준수한 기사 전문 번역)
        """

        user_prompt = f"다음 해외 뉴스 기사 원문을 분석하고 번역해 줘:\n\n{article_text[:6000]}"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.2,
            }
        )

        print("Gemini 요약 및 번역 완료!")
        return {"result_text": response.text}

    except errors.APIError as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg:
            if is_using_public_key:
                raise HTTPException(status_code=429, detail="PUBLIC_API_EXHAUSTED")
            else:
                raise HTTPException(status_code=429, detail="입력하신 개인 Gemini API 키의 사용 한도가 초과되었습니다.")
        raise HTTPException(status_code=400, detail=f"Gemini API 오류: {e}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 처리 중 에러 발생: {e}")