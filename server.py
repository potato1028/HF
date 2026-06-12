import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
from collections import Counter

# 3사 AI SDK 임포트
from openai import OpenAI
from google import genai
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
    tone: str = ""

# 간단한 명사 형태의 키워드 추출 함수 (정규식 활용)
def extract_keywords(text: str, num: int = 5):
    words = re.findall(r'\b[가-힣]{2,}\b', text)
    counter = Counter(words)
    return [word for word, count in counter.most_common(num)]

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/translate_news")
def translate_news(req: NewsRequest): 
    # 1. 뉴스 웹페이지 본문 크롤링
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

    # 2. 공통 프롬프트 세팅
    tone_instruction = f"번역 어조/타겟: {req.tone}" if req.tone.strip() else "번역 어조: 신뢰감 있는 전문 뉴스 앵커 스타일"
    system_instruction = f"""
    너는 20년 경력의 수석 해외 뉴스 전문 번역가야.
    1. 직역 금지, 문맥을 살려 자연스러운 한국어 의역.
    2. 고유명사 첫 등장 시 한글 표기와 함께 괄호 안에 영문 병기.
    3. {tone_instruction}
    
    [출력 규칙]
    반드시 아래 두 섹션으로만 나누어 한국어로 출력해. 각 섹션 제목은 굵은 글씨(**)로 표시.
    **📌 핵심 3줄 요약**
    **📰 전체 한글 번역**
    """
    user_prompt = f"다음 해외 뉴스 기사 원문을 분석하고 번역해 줘:\n\n{article_text[:6000]}"
    
    result_text = ""

    # 3. 3중 폭포수(Waterfall) AI 호출 로직
    # 1순위: OpenAI (GPT-4o-mini)
    try:
        print("1순위: OpenAI 호출 시도...")
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key: raise Exception("OpenAI Key Not Found")
        
        client = OpenAI(api_key=openai_key)
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, max_tokens=2048
        )
        result_text = res.choices[0].message.content
        print("OpenAI 번역 성공!")

    except Exception as e1:
        print(f"OpenAI 실패({e1}), 2순위: Gemini 호출 시도...")
        # 2순위: Gemini (Gemini-2.5-Flash)
        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key: raise Exception("Gemini Key Not Found")
            
            client = genai.Client(api_key=gemini_key)
            res = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config={"system_instruction": system_instruction, "temperature": 0.2}
            )
            result_text = res.text
            print("Gemini 번역 성공!")

        except Exception as e2:
            print(f"Gemini 실패({e2}), 3순위: HuggingFace Llama 호출 시도...")
            # 3순위: Llama (HuggingFace API)
            try:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token: raise Exception("HF Token Not Found")
                
                client = InferenceClient(token=hf_token)
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ]
                res = client.chat_completion(
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    messages=messages, temperature=0.2, max_tokens=2048
                )
                result_text = res.choices[0].message.content
                print("Llama 번역 성공!")

            except Exception as e3:
                print(f"Llama 실패({e3})")
                raise HTTPException(status_code=500, detail="현재 모든 AI 서버가 혼잡하여 번역을 수행할 수 없습니다. 잠시 후 다시 시도해 주세요.")

    # 4. 분석된 텍스트에서 키워드 추출 후 프론트엔드로 전달
    keywords = extract_keywords(result_text, 5)
    return {"result_text": result_text, "keywords": keywords}