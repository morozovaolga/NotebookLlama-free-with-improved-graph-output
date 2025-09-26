from dotenv import load_dotenv
import json
import os


from pydantic import BaseModel, Field, model_validator
from typing import List, Tuple, Optional
from typing_extensions import Self

load_dotenv()


class ClaimVerification(BaseModel):
    claim_is_true: bool = Field(
        description="Based on the provided sources information, the claim passes or not."
    )
    supporting_citations: Optional[List[str]] = Field(
        description="A minimum of one and a maximum of three citations from the sources supporting the claim. If the claim is not supported, please leave empty",
        default=None,
        min_length=1,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_claim_ver(self) -> Self:
        if not self.claim_is_true and self.supporting_citations is not None:
            self.supporting_citations = ["The claim was deemed false."]
        return self







from utils import ollama_chat

def verify_claim(
    claim: str,
    sources: str,
) -> Tuple[bool, Optional[List[str]]]:
    prompt = (
        f"Проверь утверждение: '{claim}'\n"
        f"Вот источники:\n{sources}\n"
        "Ответь в формате JSON: {\n 'claim_is_true': bool, 'supporting_citations': [строки] }. "
        "Если утверждение не подтверждается, supporting_citations должен быть пустым."
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = ollama_chat(messages)
    try:
        response_json = json.loads(response)
        claim_is_true = response_json.get("claim_is_true", False)
        supporting_citations = response_json.get("supporting_citations", [])
        return claim_is_true, supporting_citations
    except Exception as e:
        print(f"Ошибка парсинга ответа Ollama: {e}\nОтвет: {response}")
        return False, []
