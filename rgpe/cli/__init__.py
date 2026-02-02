from .commands import COMMANDS


def display_help():
    print("Usage: python main.py [command] [?options]")
    print("Available commands:")
    for command_key, command_cls in COMMANDS.items():
        print(f'\t{command_key} {command_cls.options()}')


def run(command_key: str, *args):
    if command_key == 'help':
        display_help()
        exit(0)

    if command_key not in COMMANDS:
        display_help()
        return

    command = COMMANDS[command_key]()
    command.execute(*args)
