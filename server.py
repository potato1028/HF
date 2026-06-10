import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

# 🌟 Hugging Face Inference API 클라이언트 임포트
from huggingface_hub import InferenceClient

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
    # 🌟 공용 API 키 처리 로직 (HF_TOKEN 환경변수 사용)
    public_hf_token = os.getenv("HF_TOKEN")
    use_public_key = False
    
    if not req.api_key.strip():
        if not public_hf_token:
            raise HTTPException(status_code=400, detail="서버에 공용 HF_TOKEN이 설정되지 않았습니다. 개인 토큼을 입력해주세요.")
        actual_token = public_hf_token
        use_public_key = True
    else:
        actual_token = req.api_key.strip()

    print(f"\n{req.url} 뉴스 기사 크롤링 시작...")

    # 1. 뉴스 웹페이지 본문 크롤링
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(req.url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
        
        if not article_text:
            raise Exception("기사 본문을 추출할 수 없습니다. 다른 뉴스 사이트 링크로 시도해 주세요.")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"뉴스 기사 가져오기 실패 : {e}")

    # 2. Llama-3-8B-Instruct 모델 호출
    try:
        print("Llama-3-8B 모델 분석 요청 중...")
        
        client = InferenceClient(token=actual_token)
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tone_instruction = f"번역 어조/타겟: {req.tone}" if req.tone.strip() else "번역 어조: 명확하고 전문적인 뉴스 앵커 스타일"

        # Llama-3 맞춤형 역할 부여 (System Prompt)
        system_prompt = f"""
        너는 최고의 AI 해외 뉴스 번역가이자 요약 에디터야.
        반드시 모든 대답은 한국어로만 작성해야 해.

        [출력 규칙]
        1. 반드시 아래 두 가지 섹션으로 나누어 한국어로 출력해.
           섹션 1: "📌 핵심 3줄 요약" (기사의 가장 중요한 내용을 3개의 불릿 포인트로 요약)
           섹션 2: "📰 전체 한글 번역" (원문의 흐름을 살려 전체 내용을 명확하게 번역)
        2. {tone_instruction}
        3. 각 섹션의 제목은 굵은 글씨(**)로 표시해.
        """

        user_prompt = f"다음 해외 뉴스 기사 원문을 분석하고 번역해 줘:\n\n{article_text[:6000]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # API 요청 보내기
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.2
        )

        result_text = response.choices[0].message.content
        print("AI 요약 및 번역 완료!")
        
        return {"result_text": result_text}

    except Exception as e:
        error_msg = str(e).lower()
        # 429나 Rate limit 에러 발생 시 공용 한도 초과 메세지 전송
        if "rate limit" in error_msg or "too many requests" in error_msg or "429" in error_msg:
            if use_public_key:
                raise HTTPException(status_code=429, detail="PUBLIC_API_EXHAUSTED")
            else:
                raise HTTPException(status_code=429, detail="입력하신 개인 HF 토큰의 호출 한도가 초과되었습니다.")
                
        raise HTTPException(status_code=500, detail=f"Llama-3 처리 중 에러 발생 : {e}")