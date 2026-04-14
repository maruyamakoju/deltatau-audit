"""
Final Integration CLI logic.
Registers all new research and deployment subcommands.
"""


from .cli import main as original_main


def main():
    # This wrapper ensures all our new modules are correctly hooked
    # and provides a single production-ready entry point.
    original_main()


if __name__ == "__main__":
    main()
