import csv
from math import log, pi


def as_exp_theta(x: float, n: int) -> float:
    return (x / 2) * log(x / (2 * pi)) - x / 2 - pi / 8 + 1 / (48 * x) + 7 / (5760 * x ** 3) - pi * n + pi


def numerical_solution(seed: float, prec: float, n: int) -> float:
    return numerical_solution_recursion(seed, prec, 100.0, n)


def numerical_solution_recursion(curr_x: float, prec: float, step: float, n: int) -> float:
    cur_val = as_exp_theta(curr_x, n)
    next_val_plus = as_exp_theta(curr_x + step, n)
    next_val_minus = as_exp_theta(max(curr_x - step, 7.0), n)

    while (next_val_plus * cur_val > 0) and (next_val_minus * cur_val > 0):
        if curr_x - step >= 7.0:
            if abs(next_val_plus) <= abs(next_val_minus):
                curr_x += step
            else:
                curr_x -= step
        else:
            if abs(next_val_plus) >= abs(next_val_minus):
                curr_x = 7.0
            else:
                curr_x += step

        cur_val = as_exp_theta(curr_x, n)
        next_val_plus = as_exp_theta(curr_x + step, n)
        next_val_minus = as_exp_theta(max(curr_x - step, 7.0), n)

    if step <= prec and ((next_val_minus * cur_val <= 0) or (next_val_plus * cur_val <= 0)):
        return curr_x

    if next_val_plus * cur_val < 0:
        return numerical_solution_recursion(curr_x + step, prec, step / 2, n)
    elif curr_x - step >= 7.0 and next_val_minus * cur_val < 0:
        return numerical_solution_recursion(curr_x - step, prec, step / 2, n)
    elif curr_x - step < 7.0 and next_val_minus * cur_val < 0:
        return numerical_solution_recursion(7.0, prec, step / 2, n)


def write_gram_points(filename: str, start: int, end: int, seed: float = 8, prec: float = 1e-7) -> None:
    gram_fields = ['n', 'n-th gram point']
    with open(filename, 'w', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=gram_fields)
        writer.writeheader()
        for x in range(start, end):
            if x % 1000 == 0:
                print(f"Progress: {x / (end - start) * 100:.2f}%")
            gram_point = numerical_solution(seed, prec, x)
            writer.writerow({'n': x, 'n-th gram point': gram_point})
