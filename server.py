import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# 🌟 새로운 Gemini SDK 임포트
from google import genai
from google.genai import errors

app = FastAPI()

class NewsRequest(BaseModel):
    url: str
    api_key: str = "" # 사용자가 입력한 API 키 (선택 사항)
    tone: str = ""

@app.post("/api/translate_news")
def translate_news(req: NewsRequest):
    # ---------------------------------------------------------
    # 1. API 키 결정 로직 (사용자 키 vs 공용 키)
    # ---------------------------------------------------------
    server_public_gemini_key = os.getenv("GEMINI_API_KEY")
    is_using_public_key = False
    
    if req.api_key.strip():
        # 사용자가 키를 입력한 경우
        final_api_key = req.api_key.strip()
    else:
        # 사용자가 키를 입력하지 않은 경우 -> 공용 키 사용
        if not server_public_gemini_key:
            raise HTTPException(status_code=400, detail="서버 공용 토큰이 설정되지 않았습니다. 개인 Gemini API 키를 입력해 주세요.")
        final_api_key = server_public_gemini_key
        is_using_public_key = True

    # (뉴스 크롤링 로직은 기존과 동일하게 유지...)
    article_text = "크롤링된 뉴스 본문 예시..." 

    # ---------------------------------------------------------
    # 2. Gemini API로 프롬프트 전송 및 응답 받기
    # ---------------------------------------------------------
    try:
        print("Gemini 모델 분석 요청 중...")
        
        # 결정된 API 키로 Gemini 클라이언트 초기화
        client = genai.Client(api_key=final_api_key)
        
        # 시스템 프롬프트(번역 가이드라인) 및 유저 프롬프트 설정
        system_instruction = "너는 20년 경력의 수석 해외 뉴스 전문 번역가야. 직역을 피하고 자연스럽게 의역해."
        user_prompt = f"다음 기사를 분석하고 요약 및 번역해줘:\n\n{article_text[:6000]}"

        # Gemini 2.5 Flash 모델 호출 (빠르고 가벼운 모델)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.2,
            }
        )

        result_text = response.text
        print("Gemini 요약 및 번역 완료!")
        
        return {"result_text": result_text}

    except errors.APIError as e:
        # API 오류 (예: 할당량 초과, 잘못된 키 등) 처리
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg:
            if is_using_public_key:
                raise HTTPException(status_code=429, detail="PUBLIC_API_EXHAUSTED") # 프론트엔드 배너 경고용
            else:
                raise HTTPException(status_code=429, detail="입력하신 개인 Gemini API 키의 사용 한도가 초과되었습니다.")
        raise HTTPException(status_code=400, detail=f"Gemini API 오류: {e}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 에러 발생: {e}")