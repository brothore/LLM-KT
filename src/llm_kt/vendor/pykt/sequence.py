"""Adapted from pykt/preprocess/split_datasets_que.py generate_sequences/window.

Copyright (c) 2022 pykt-team. MIT; see third_party/pykt/LICENSE.
Array/dataframe generation is reduced to pure target/history index ranges.
"""


def target_ranges(length, protocol, maxlen=200, min_seq_len=3):
    if protocol == "full_prefix":
        for index in range(1, length):
            yield 0, index
    elif protocol == "pykt_window":
        # First window predicts all but the first item; later windows only their last.
        for index in range(1, length):
            yield max(0, index - maxlen + 1), index
    elif protocol == "pykt_segmented":
        for start in range(0, length, maxlen):
            end = min(start + maxlen, length)
            if end - start < min_seq_len:
                continue
            for index in range(start + 1, end):
                yield start, index
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
