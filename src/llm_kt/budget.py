"""Durable conservative reservations shared across sequential experiment runs."""
from decimal import Decimal
import math
from pathlib import Path
from .common import canonical, read_json, write_json
from .artifacts import utc_now


class Budget:
    def __init__(self, path, limit='0.40'):
        self.path = Path(path)
        self.data = read_json(path) if self.path.exists() else {'limit_usd': str(limit), 'attempts': {}}
        if Decimal(self.data['limit_usd']) != Decimal(limit):
            raise ValueError('Budget limit changed')

    def totals(self):
        known = sum((Decimal(str(r['cost'])) for r in self.data['attempts'].values() if r.get('cost') is not None), Decimal(0))
        reserved = sum((Decimal(r['reserved']) for r in self.data['attempts'].values() if r.get('cost') is None), Decimal(0))
        return known, reserved

    def reserve(self, request_id, payload, model):
        rates = model['pricing_snapshot']['endpoints']
        allowed = model['parameters'].get('provider', {}).get('only', [])
        rates = [r for r in rates if r['tag'] in allowed]
        if len(rates) != 1:
            raise ValueError('Budgeted run requires exactly one priced provider')
        row = rates[0]
        ip = max(row['input_per_million_usd'], row.get('cache_write_per_million_usd') or 0)
        amount = (Decimal(len(canonical(payload).encode()) + 4096) * Decimal(str(ip))
                  + Decimal(payload.get('max_tokens', 128)) * Decimal(str(row['output_per_million_usd']))) / Decimal(1000000) * 4
        known, reserved = self.totals()
        if known + reserved + amount > Decimal(self.data['limit_usd']):
            raise ValueError('Experiment budget would be exceeded')
        if request_id in self.data['attempts']:
            raise ValueError('Duplicate budget attempt')
        self.data['attempts'][request_id] = {'reserved': str(amount), 'cost': None, 'at': utc_now()}
        write_json(self.path, self.data)

    def settle(self, request_id, cost):
        if isinstance(cost, (float, int)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
            self.data['attempts'][request_id]['cost'] = cost
        write_json(self.path, self.data)
