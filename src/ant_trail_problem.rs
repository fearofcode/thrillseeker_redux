use crate::ant_trail::{
    Direction, Grid, WorldPosition, LOS_ALTOS_PERFECT_SCORE, MAXIMUM_MOVEMENTS,
};
use crate::{BehaviorDescriptor, Function, ProblemParameters, RunParameters, Team};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Serialize, Deserialize, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AntTrailAction {
    Up,
    Down,
    Left,
    Right,
    // UpRight,
    // DownRight,
    // UpLeft,
    // DownLeft,
}

pub fn index_to_ant_trail_action(index: usize) -> AntTrailAction {
    match index {
        0 => AntTrailAction::Up,
        1 => AntTrailAction::Down,
        2 => AntTrailAction::Left,
        3 => AntTrailAction::Right,
        // 4 => AntTrailAction::UpRight,
        // 5 => AntTrailAction::DownRight,
        // 6 => AntTrailAction::UpLeft,
        // 7 => AntTrailAction::DownLeft,
        _ => panic!(),
    }
}

impl fmt::Display for AntTrailAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AntTrailAction::Up => write!(f, "Up"),
            AntTrailAction::Down => write!(f, "Down"),
            AntTrailAction::Left => write!(f, "Left"),
            AntTrailAction::Right => write!(f, "Right"),
            // AntTrailAction::UpRight => write!(f, "UpRight"),
            // AntTrailAction::DownRight => write!(f, "DownRight"),
            // AntTrailAction::UpLeft => write!(f, "UpLeft"),
            // AntTrailAction::DownLeft => write!(f, "DownLeft"),
        }
    }
}

#[derive(Serialize, Deserialize, Hash, Eq, PartialEq, Debug, Copy, Clone)]
pub struct AntTrailFitness {
    pub food_remaining: usize,
    pub steps_taken: usize,
}

impl Display for AntTrailFitness {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.food_remaining, self.steps_taken)
    }
}

impl PartialOrd<Self> for AntTrailFitness {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AntTrailFitness {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.food_remaining < other.food_remaining {
            return Ordering::Less;
        }
        if self.food_remaining == other.food_remaining && self.steps_taken < other.steps_taken {
            return Ordering::Less;
        }
        if self.food_remaining == other.food_remaining && self.steps_taken == other.steps_taken {
            return Ordering::Equal;
        }

        Ordering::Greater
    }
}

pub fn ant_trail_individual_output(
    team: &Team<AntTrailAction, AntTrailFitness>,
    _fitness_cases: &[Vec<f32>],
    params: &ProblemParameters<AntTrailFitness>,
    _unused_labels: &[AntTrailAction],
) -> (AntTrailFitness, BehaviorDescriptor) {
    let (food_gathered, steps_taken, behavior) = simulate_ant_trail(team, params, None);

    let fitness = AntTrailFitness {
        food_remaining: (LOS_ALTOS_PERFECT_SCORE - food_gathered),
        steps_taken,
    };

    (fitness, behavior)
}

pub fn simulate_ant_trail(
    team: &Team<AntTrailAction, AntTrailFitness>,
    params: &ProblemParameters<AntTrailFitness>,
    callback: Option<fn(Grid, WorldPosition)>,
) -> (usize, usize, BehaviorDescriptor) {
    let mut moves = vec![];
    let mut pos = WorldPosition::new();

    let mut movement_count = 0;

    let mut food_gathered = 0;
    let mut grid = Grid::los_altos_trail();

    if grid.food_at_position(pos) {
        food_gathered += 1;
        grid.remove_food_at_position(pos);
    }

    while movement_count < MAXIMUM_MOVEMENTS {
        let mut state: Vec<f32> = vec![
            if grid.is_food_in_direction(pos, Direction::Up) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::Down) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::Left) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::Right) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::UpRight) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::DownRight) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::UpLeft) {
                1.0
            } else {
                0.0
            },
            if grid.is_food_in_direction(pos, Direction::DownLeft) {
                1.0
            } else {
                0.0
            },
        ];
        for c in params.constant_list.iter() {
            state.push(*c);
        }
        let outputs = crate::evaluate_team(team, &[state], params);
        let output = outputs[0];
        moves.push(match output {
            AntTrailAction::Up => "U",
            AntTrailAction::Down => "D",
            AntTrailAction::Left => "L",
            AntTrailAction::Right => "R",
            // AntTrailAction::UpRight => {
            //     pos.facing = Direction::UpRight;
            // }
            // AntTrailAction::DownRight => {
            //     pos.facing = Direction::DownRight;
            // }
            // AntTrailAction::UpLeft => {
            //     pos.facing = Direction::UpLeft;
            // }
            // AntTrailAction::DownLeft => {
            //     pos.facing = Direction::DownLeft;
            // }
        });

        match output {
            AntTrailAction::Up => {
                pos.facing = Direction::Up;
            }
            AntTrailAction::Down => {
                pos.facing = Direction::Down;
            }
            AntTrailAction::Left => {
                pos.facing = Direction::Left;
            }
            AntTrailAction::Right => {
                pos.facing = Direction::Right;
            } // AntTrailAction::UpRight => {
              //     pos.facing = Direction::UpRight;
              // }
              // AntTrailAction::DownRight => {
              //     pos.facing = Direction::DownRight;
              // }
              // AntTrailAction::UpLeft => {
              //     pos.facing = Direction::UpLeft;
              // }
              // AntTrailAction::DownLeft => {
              //     pos.facing = Direction::DownLeft;
              // }
        }

        pos.one_move();

        if grid.food_at_position(pos) {
            food_gathered += 1;
            if food_gathered >= LOS_ALTOS_PERFECT_SCORE {
                return (food_gathered, movement_count, moves.join(""));
            }
            grid.remove_food_at_position(pos);
        }

        if let Some(cb) = callback {
            cb(grid, pos);
        }

        movement_count += 1;
    }
    (food_gathered, movement_count, moves.join(""))
}

pub fn ant_trail_parameters() -> ProblemParameters<AntTrailFitness> {
    ProblemParameters {
        input_count: 8,
        register_count: 4,
        population_size: 50000,
        keep_by_fitness: 100,
        keep_by_novelty: 1000,
        select_by_novelty: 25000,
        max_program_size: 64,
        min_initial_program_size: 1,
        max_initial_program_size: 8,
        // Up, Down, Left, Right
        action_count: 4,
        max_initial_team_size: 8,
        max_team_size: 32,
        tournament_size: 4,
        generation_count: 1000,
        generation_stagnation_limit: 25,
        run_count: 1,
        p_delete_instruction: 0.7,
        p_add_instruction: 0.7,
        p_swap_instructions: 0.7,
        p_change_destination: 0.7,
        p_change_function: 0.7,
        p_change_input: 0.7,
        p_flip_input: 0.7,
        p_change_action: 0.7,
        p_delete_program: 0.7,
        p_add_program: 0.7,
        fitness_threshold: AntTrailFitness {
            food_remaining: 0,
            steps_taken: 300,
        },
        legal_functions: vec![
            // Function::Relu,
            // Function::Plus,
            // Function::Minus,
            // Function::Times,
            // Function::Divide,
            // Function::Square,
            // Function::Sin,
            // Function::Log,
            Function::And,
            Function::Or,
            Function::Not,
            Function::Xor,
            // Function::Min,
            // Function::Max,
            // Function::Greater,
            // Function::Less,
            Function::IfThenElse,
            // Function::Copy,
        ],
        constant_list: vec![0.0, 1.0, -1.0],
        feature_names: vec![
            "food-at-up",
            "food-at-down",
            "food-at-left",
            "food-at-right",
            "food-at-up-right",
            "food-at-down-right",
            "food-at-up-left",
            "food-at-down-left",
        ],
    }
}

pub fn ant_trail_runs(
    seed: u64,
    dump: bool,
    rng: &mut Rng,
) -> (
    Vec<Team<AntTrailAction, AntTrailFitness>>,
    ProblemParameters<AntTrailFitness>,
) {
    let mut id_counter: usize = 1;

    let mut best_teams: Vec<Team<AntTrailAction, AntTrailFitness>> = vec![];

    let params = ant_trail_parameters();

    for run in 1..=params.run_count {
        let mut run_parameters: RunParameters<AntTrailAction, AntTrailFitness> = RunParameters {
            run,
            rng,
            fitness_cases: &[],
            labels: &[],
            problem_parameters: &params,
            individual_output: ant_trail_individual_output,
            index_to_program_action: index_to_ant_trail_action,
            id_counter: &mut id_counter,
            dump,
            seed,
        };

        best_teams.push(crate::one_run(&mut run_parameters));
    }
    (best_teams, params)
}
