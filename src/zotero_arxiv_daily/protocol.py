from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')


def _request_llm(openai_client: OpenAI, llm_params: dict, messages: list[dict]) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in generation_kwargs:
            generation_kwargs["max_output_tokens"] = max_tokens
        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    translated_abstract: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> tuple[str, Optional[str]]:
        lang = llm_params.get('language', 'English')
        prompt = f"""Given the following information of a scientific paper, produce both a concise TLDR and a Chinese translation of the abstract.

Return ONLY a valid JSON object with exactly these keys:
{{
  \"tldr\": \"...\",
  \"chinese_abstract\": \"...\"
}}

Requirements:
1. \"tldr\": Write one concise sentence in {lang}. State the paper's main problem, method, and core contribution whenever the source text supports them.
2. \"chinese_abstract\": Faithfully translate the COMPLETE provided Abstract into fluent Simplified Chinese. Do not summarize, shorten, or omit claims. Preserve method names, model names, datasets, benchmarks, technical terms, numbers, and experimental conclusions. If a technical term is clearer in English, keep the English term in parentheses after the Chinese translation.
3. Do not invent information that is not present in the paper.
4. If no Abstract is provided, set \"chinese_abstract\" to an empty string.

"""
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided", None
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response_text = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who accurately summarizes scientific papers and "
                        "faithfully translates scientific abstracts into Simplified Chinese. "
                        "Follow the requested JSON output format exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        try:
            json_match = re.search(r'\{.*\}', response_text, flags=re.DOTALL)
            if json_match is None:
                raise ValueError("No JSON object found in LLM response")
            result = json.loads(json_match.group(0))
            tldr = str(result["tldr"]).strip()
            translated_abstract = str(result.get("chinese_abstract", "")).strip() or None
            return tldr, translated_abstract
        except Exception as e:
            logger.warning(f"Failed to parse structured TLDR/translation for {self.url}: {e}")
            return response_text.strip(), None
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr, translated_abstract = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            self.translated_abstract = translated_abstract
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            self.translated_abstract = None
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]