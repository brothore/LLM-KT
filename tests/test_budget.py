import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os
from llm_kt.budget import Budget
from llm_kt.common import write_json, read_json


class BudgetTests(unittest.TestCase):
    def test_persisted_unknown_charge_is_not_released(self):
        model={'pricing_snapshot':{'endpoints':[{'tag':'synthetic','input_per_million_usd':.02,'output_per_million_usd':.5}]},
               'parameters':{'provider':{'only':['synthetic']}}}
        with patch('llm_kt.budget.Path.exists',return_value=False),patch('llm_kt.budget.write_json'):
            budget=Budget('synthetic/budget.json')
            budget.reserve('one',{'max_tokens':128},model)
            before=budget.totals()[1]
            budget.settle('one',None)
            self.assertEqual(budget.totals()[1],before)
            budget.settle('one',.00001)
            self.assertEqual(budget.totals()[1],0)
            self.assertGreater(budget.totals()[0],0)

    def test_atomic_json_retries_transient_windows_sharing_violation(self):
        # Pure mock avoids host-specific filesystem ACL behavior.
        with (patch('llm_kt.common.Path.mkdir'),patch('llm_kt.common.Path.write_text'),patch('llm_kt.common.time.sleep') as sleep,
              patch('llm_kt.common.os.replace',side_effect=[PermissionError('busy'),None]) as replace):
            write_json('synthetic/test.json',{'a':1})
            self.assertEqual(replace.call_count,2)
            sleep.assert_called_once()
