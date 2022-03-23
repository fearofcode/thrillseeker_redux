// modified from https://github.com/fearofcode/ant_trail/blob/master/src/lib.rs

static SANTA_FE_ANT_TRAIL: &str =
    ".###............................
...#............................
...#.....................###....
...#....................#....#..
...#....................#....#..
...####.#####........##.........
............#................#..
............#.......#...........
............#.......#........#..
............#.......#...........
....................#...........
............#................#..
............#...................
............#.......#.....###...
............#.......#..#........
.................#..............
................................
............#...........#.......
............#...#..........#....
............#...#...............
............#...#...............
............#...#.........#.....
............#..........#........
............#...................
...##..#####....#...............
.#..............#...............
.#..............#...............
.#......#######.................
.#.....#........................
.......#........................
..####..........................
................................";

// number of #'s above. we can terminate if this score is achieved
pub const SANTA_FE_PERFECT_SCORE: usize = 89;

pub const MAXIMUM_MOVEMENTS: usize = GRID_SIZE_USIZE*GRID_SIZE_USIZE;


// assume every grid is this size
pub const GRID_SIZE: i8 = 32;
pub const GRID_SIZE_USIZE: usize = GRID_SIZE as usize;

#[derive(PartialEq, Copy, Debug, Clone)]
pub struct Grid {
    // not as cache-efficient as a bitset, but easier to implement toroidal movement on
    pub grid: [[bool; GRID_SIZE_USIZE]; GRID_SIZE_USIZE]
}

impl Grid {
    pub fn from_trail_string(trail_str: &str) -> Grid {
        let mut grid = Grid { grid: [[false; GRID_SIZE_USIZE]; GRID_SIZE_USIZE] };

        let mut max_row_idx = 0;

        for (row_idx, line) in trail_str.split_ascii_whitespace().enumerate() {
            if line.len() != GRID_SIZE_USIZE {
                panic!("Invalid line length");
            }

            for (col_idx, ch) in line.chars().enumerate() {
                match ch {
                    '#' => {
                        grid.grid[row_idx][col_idx] = true;
                    },
                    '.' => { },
                    _ => {
                        panic!("invalid char");
                    }
                }
            }
            max_row_idx = row_idx;
        }

        if max_row_idx != GRID_SIZE_USIZE - 1 {
            panic!("Invalid row count");
        }

        grid
    }

    pub fn santa_fe_trail() -> Grid {
        Grid::from_trail_string(SANTA_FE_ANT_TRAIL)
    }

    pub fn draw(&self) {
        for row in &self.grid {
            for col in row {
                if *col {
                    print!("#");
                } else {
                    print!(".");
                }
            }
            println!();
        }
    }

    pub fn draw_with_position(&self, pos: WorldPosition) {
        let pos_row = pos.y as usize;
        let pos_col = pos.x as usize;

        for (row_idx, row) in self.grid.iter().enumerate() {
            for (col_idx, col) in row.iter().enumerate() {
                if row_idx == pos_row && col_idx == pos_col {
                    // ant
                    print!("A");
                } else if *col {
                    print!("#");
                } else {
                    print!(".");
                }
            }
            println!();
        }
    }

    pub fn food_at_position(&self, pos: WorldPosition) -> bool {
        self.grid[pos.y as usize][pos.x as usize]
    }

    pub fn remove_food_at_position(&mut self, pos: WorldPosition) {
        self.grid[pos.y as usize][pos.x as usize] = false;
    }

    pub fn is_food_in_direction(&self, pos: WorldPosition, direction: Direction) -> bool {
        let next_pos = pos.position_ahead(direction);
        self.food_at_position(next_pos)
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpRight,
    DownRight,
    UpLeft,
    DownLeft
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub struct WorldPosition {
    // makes implementing movement easy
    x: i8,
    y: i8,
    pub facing: Direction
}

impl WorldPosition {
    pub fn new() -> WorldPosition {
        WorldPosition { x: 0, y: 0, facing: Direction::Right}
    }

    fn new_from_coords(x: i8, y: i8, facing: Direction) -> Option<WorldPosition> {
        if x < GRID_SIZE && y < GRID_SIZE {
            Some(WorldPosition { x, y, facing})
        } else {
            None
        }
    }

    pub fn position_ahead(self, dir: Direction) -> WorldPosition {
        let mut new_x = self.x + match dir {
            Direction::Down => 0,
            Direction::Left => -1,
            Direction::Right => 1,
            Direction::Up => 0,
            Direction::UpRight => 1,
            Direction::DownRight => 1,
            Direction::UpLeft => -1,
            Direction::DownLeft => -1
        };

        // toroidal world: moving off the grid moves you back on to the grid on the other side (it's legal)
        if new_x < 0 {
            new_x += GRID_SIZE;
        } else if new_x > GRID_SIZE - 1 {
            new_x -= GRID_SIZE;
        }

        let mut new_y = self.y + match dir {
            Direction::Down => 1,
            Direction::Left => 0,
            Direction::Right => 0,
            Direction::Up => -1,
            Direction::UpRight => -1,
            Direction::DownRight => 1,
            Direction::UpLeft => -1,
            Direction::DownLeft => 1
        };

        if new_y < 0 {
            new_y += GRID_SIZE;
        } else if new_y > GRID_SIZE - 1 {
            new_y -= GRID_SIZE;
        }

        WorldPosition::new_from_coords(new_x, new_y, self.facing).unwrap()
    }

    pub fn one_move(&mut self) {
        *self = self.position_ahead(self.facing)
    }
}