"""Run one fully traceable prompt experiment under a shared spend ledger."""
import argparse
import asyncio
from pathlib import Path
from llm_kt.budget import Budget
from llm_kt.config import load_config
from llm_kt.runtime import infer, verify_run
from llm_kt.evaluation import evaluate
from llm_kt.common import read_json, write_json


async def run(args):
    config = load_config(args.config)
    budget = Budget(args.budget, '0.40')
    summary = await infer(config, args.prepared, args.run, budget, args.allow_code_change)
    report = evaluate(args.prepared, args.run)
    verify_run(args.run)
    known, reserved = budget.totals()
    result = {'run': str(Path(args.run).resolve()), 'execution': summary, 'evaluation': report,
              'experiment_budget_known_usd': str(known), 'experiment_budget_reserved_usd': str(reserved)}
    write_json(Path(args.run) / 'experiment_summary.json', result)
    print('RESULT', result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--budget', default='runs/prompt_iterations_20261018/budget.json')
    p.add_argument('--allow-code-change', action='store_true', help='Record explicit source transition after infrastructure recovery; data/config must match')
    args = p.parse_args()
    asyncio.run(run(args))
