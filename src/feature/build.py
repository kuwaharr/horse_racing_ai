def is_place(finish: int, race_size: int) -> int:
    if race_size <= 7:
        return int(finish <= 2)
    return int(finish <= 3)