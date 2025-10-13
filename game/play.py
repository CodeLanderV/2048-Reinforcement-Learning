"""
Main entry point to play 2048 game
"""

from game_ui import Game2048UI


def main():
    """Launch the 2048 game"""
    print("Starting 2048 game...")
    print("Controls:")
    print("  Arrow keys - Move tiles")
    print("  R - Restart game")
    print("  ESC - Quit")
    print()
    
    game = Game2048UI()
    game.run()


if __name__ == "__main__":
    main()
