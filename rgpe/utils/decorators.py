from typing import Callable


def log_dataset_generation_execution(dataset_name: str) -> Callable:
    def decorator(func: Callable[[int], None]) -> Callable:
        def wrapper(*args, **kwargs):
            print(f"Generating {dataset_name} dataset...")
            func(*args, **kwargs)
            print(f"[OK] {dataset_name} dataset generated successfully.")

        return wrapper

    return decorator
