PER_QUERY = 3
NUM_QUERIES = 36
NUM_LEARNING_BREAKS = 2

PER_EVAL = 3
NUM_EVAL = 6
NUM_EVAL_BREAKS = 1
DOUBLE_REVERSAL = True

BREAK_TIME = 5
BUFFER = 1.1

LEARNING_TIME = PER_QUERY * NUM_QUERIES + BREAK_TIME * NUM_LEARNING_BREAKS

# A double reversal runs every evaluation trial twice.
EVAL_TIME = PER_EVAL * NUM_EVAL * (2 if DOUBLE_REVERSAL else 1) + BREAK_TIME * NUM_EVAL_BREAKS

TOTAL_TIME = LEARNING_TIME + EVAL_TIME


def fmt(minutes):
    hours = int(minutes // 60)
    return f'{hours} hr {minutes - hours * 60:4.1f} min'


def phase(name, total, num_breaks):
    """Prints a phase's total time and the working time between its breaks."""
    print(f'{name}\n=====')
    print(f'  total            {fmt(total)}')
    print(f'  between breaks   {fmt((total - num_breaks * BREAK_TIME) / (num_breaks + 1))}')
    print(f'  breaks           {num_breaks} x {BREAK_TIME} min')
    print()


print()
phase('Learning phase', LEARNING_TIME, NUM_LEARNING_BREAKS)
phase('Evaluation phase', EVAL_TIME, NUM_EVAL_BREAKS)

print('Totals\n=====')
print(f'  learning + eval  {fmt(TOTAL_TIME)}')
print(f'  with buffer      {fmt(TOTAL_TIME * BUFFER)}  ({BUFFER:.0%})')
print()
