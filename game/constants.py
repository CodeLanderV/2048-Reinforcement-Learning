"""
Constants for 2048 game
"""

# Board settings
GRID_SIZE = 4
TILE_SIZE = 100
GRID_PADDING = 10

# Window settings
WINDOW_WIDTH = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * GRID_PADDING
WINDOW_HEIGHT = WINDOW_WIDTH + 100  # Extra space for score
FPS = 60

# Colors (R, G, B)
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_CELL_COLOR = (205, 193, 180)
SCORE_BG_COLOR = (238, 228, 218)
TEXT_COLOR = (119, 110, 101)
TEXT_COLOR_LIGHT = (249, 246, 242)

# Tile colors
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
}

# Font sizes
FONT_SIZE_SMALL = 24
FONT_SIZE_MEDIUM = 36
FONT_SIZE_LARGE = 55
FONT_SIZE_TILE = {
    2: 55,
    4: 55,
    8: 55,
    16: 50,
    32: 50,
    64: 50,
    128: 45,
    256: 45,
    512: 45,
    1024: 35,
    2048: 35,
    4096: 30,
    8192: 30,
}

# Animation
ANIMATION_SPEED = 10
