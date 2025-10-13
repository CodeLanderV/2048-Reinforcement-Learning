"""
Main entry point to play 2048 game
"""

from game_ui import Game2048UI


def main():
    game = Game2048UI()
    game.run()


if __name__ == "__main__":
    main()
