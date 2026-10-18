import json
import re
from .common import canonical
from .domain import checked_probability

TASK = ("Predict this student's probability of answering the current item correctly, based only on observed history. "
        "IDs are identifiers, not item contents. Do not solve an imagined item. Return only a JSON object with p_correct, a number from 0 to 1.")


class Decisions:
    def __init__(self, config):
        self.config = config

    def request(self, item):
        strategy = self.config["probability_strategy"]
        if self.config.get('decision_snapshot'):
            from .decision_prompts import render_decision
            state,question=render_decision(item,self.config['decision_snapshot'],strategy)
            return {'model':self.config['model_id'],'state':state,
                    'questions':{self.config['decision_snapshot'].get('question_key','will_answer_correctly'):question},**self.config['parameters']}
        criteria = {"true": "The student answers correctly", "false": "The student answers incorrectly"} if strategy == "noul" else {"correct": "The student answers correctly", "incorrect": "The student answers incorrectly"}
        return {"model": self.config["model_id"], "state": item.state(),
                "questions": {"will_answer_correctly": {"type": strategy,
                    "instructions": "Based only on observed history and current item identifiers, will this student answer the current item correctly?", "criteria": criteria}},
                **self.config["parameters"]}

    def probability(self, response):
        answer = response["answers"][self.config.get('decision_snapshot',{}).get('question_key','will_answer_correctly')]
        strategy = self.config["probability_strategy"]
        if answer["type"] != strategy:
            raise ValueError("Unexpected decision answer type")
        probability=checked_probability(answer["noul"] if strategy == "noul" else answer["probabilities"]["correct"])
        if strategy=='noul' and self.config.get('decision_snapshot',{}).get('positive_class')=='incorrect':
            return checked_probability(1-probability)
        return probability


class Chat:
    def __init__(self, config):
        self.config = config

    def request(self, item):
        if self.config.get('prompt_snapshot'):
            from .prompts import render
            return {'model': self.config['model_id'], 'messages': render(item, self.config['prompt_snapshot']), **self.config['parameters']}
        return {"model": self.config["model_id"], "messages": [
            {"role": "system", "content": TASK}, {"role": "user", "content": canonical(item.state())}],
            **self.config["parameters"]}

    def probability(self, response):
        choice = response["choices"][0]
        if choice.get("finish_reason") not in ("stop",):
            raise ValueError("Completion did not finish normally")
        message = choice["message"]
        if message.get("refusal"):
            raise ValueError("Model refused")
        text = message['content'].strip()
        if self.config.get('probability_strategy') == 'decimal_probability':
            if not re.fullmatch(r'(?:0(?:\.[0-9]{1,6})?|1(?:\.0{1,6})?)',text):
                raise ValueError('Expected a bare decimal probability')
            return checked_probability(float(text))
        if self.config.get('probability_strategy') == 'scaled_integer_probability':
            if not re.fullmatch(r'(?:0|[1-9][0-9]{0,3}|10000)', text):
                raise ValueError('Expected a bare integer from 0 to 10000')
            return checked_probability(int(text) / 10000)
        fenced = re.fullmatch(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        content = json.loads(fenced.group(1) if fenced else text)
        if not isinstance(content, dict) or set(content) != {"p_correct"}:
            raise ValueError("Expected exactly one p_correct field")
        return checked_probability(content["p_correct"])


class Mock:
    def request(self, item):
        return {"model": "mock-history-frequency-v1", "state": item.state()}

    def probability(self, response):
        return checked_probability(response["p_correct"])


REGISTRY = {"openrouter_decisions": Decisions, "openai_compatible_chat": Chat}


def make_provider(config):
    return Mock() if config["adapter"] == "mock" else REGISTRY[config["adapter"]](config)
