"""Sequential fresh control and identical-payload repeat under the existing budget."""
import asyncio
from run_r2_cost_optimization import main


async def controls():
    await main(['round2'],8,128)
    await main(['cost_r2_rows_repeat'],8,16,decimal_output=True)


if __name__=='__main__':
    asyncio.run(controls())
