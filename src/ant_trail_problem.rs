use crate::{Function, ProblemParameters, Team};
use fastrand::Rng;
use std::fmt;
use crate::ant_trail::{Direction, Grid, MAXIMUM_MOVEMENTS, SANTA_FE_PERFECT_SCORE, WorldPosition};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AntTrailAction {
    Up,
    Down,
    Left,
    Right,
    UpRight,
    DownRight,
    UpLeft,
    DownLeft
}

pub fn index_to_ant_trail_action(index: usize) -> AntTrailAction {
    match index {
        0 => AntTrailAction::Up,
        1 => AntTrailAction::Down,
        2 => AntTrailAction::Left,
        3 => AntTrailAction::Right,
        4 => AntTrailAction::UpRight,
        5 => AntTrailAction::DownRight,
        6 => AntTrailAction::UpLeft,
        7 => AntTrailAction::DownLeft,
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
            AntTrailAction::UpRight => write!(f, "UpRight"),
            AntTrailAction::DownRight => write!(f, "DownRight"),
            AntTrailAction::UpLeft => write!(f, "UpLeft"),
            AntTrailAction::DownLeft => write!(f, "DownLeft"),
        }
    }
}

pub fn ant_trail_individual_error(
    team: &Team<AntTrailAction>,
    _fitness_cases: &[Vec<f32>],
    params: &ProblemParameters,
    _unused_labels: &[AntTrailAction],
) -> f32 {
    let mut pos = WorldPosition::new();

    let mut movement_count = 0;

    let mut food_gathered = 0;
    let mut grid = Grid::santa_fe_trail();

    if grid.food_at_position(pos) {
        food_gathered += 1;
        grid.remove_food_at_position(pos);
    }

    while movement_count < MAXIMUM_MOVEMENTS {
        let state: Vec<f32> = vec![
            if grid.is_food_in_direction(pos, Direction::Up) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::Down) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::Left) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::Right) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::UpRight) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::DownRight) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::UpLeft) { 1.0 } else { 0.0 },
            if grid.is_food_in_direction(pos, Direction::DownLeft) { 1.0 } else { 0.0 }
        ];
        let outputs = crate::evaluate_team(team, &[state], params);
        let output = outputs[0];
        match output {
            AntTrailAction::Up => {
              pos.facing = Direction::Up;
            },
            AntTrailAction::Down => {
                pos.facing = Direction::Down;
            },
            AntTrailAction::Left => {
                pos.facing = Direction::Left;
            },
            AntTrailAction::Right => {
                pos.facing = Direction::Right;
            },
            AntTrailAction::UpRight => {
                pos.facing = Direction::UpRight;
            },
            AntTrailAction::DownRight => {
                pos.facing = Direction::DownRight;
            },
            AntTrailAction::UpLeft => {
                pos.facing = Direction::UpLeft;
            },
            AntTrailAction::DownLeft => {
                pos.facing = Direction::DownLeft;
            },
        }

        pos.one_move();

        if grid.food_at_position(pos) {
            food_gathered += 1;
            if food_gathered >= SANTA_FE_PERFECT_SCORE {
                return 0.0;
            }
            grid.remove_food_at_position(pos);
        }
        movement_count += 1;
    }

    (SANTA_FE_PERFECT_SCORE - food_gathered) as f32
}

pub fn ant_trail_runs(seed: u64, dump: bool, mut rng: &mut Rng) -> (Vec<Team<AntTrailAction>>, ProblemParameters) {
    let mut id_counter: u64 = 1;

    let mut best_teams: Vec<Team<AntTrailAction>> = vec![];

    let ant_trail_parameters = ProblemParameters {
        input_count: 4,
        register_count: 4,
        population_size: 10000,
        population_to_delete: 9500,
        max_program_size: 64,
        min_initial_program_size: 1,
        max_initial_program_size: 16,
        // Up, Down, Left, Right, UpLeft, DownLeft, UpRight, DownRight
        action_count: 8,
        max_initial_team_size: 16,
        max_team_size: 64,
        tournament_size: 4,
        generation_count: 1000,
        generation_stagnation_limit: 10,
        run_count: 1,
        p_delete_instruction: 0.7,
        p_add_instruction: 0.7,
        p_swap_instructions: 0.7,
        p_change_destination: 0.7,
        p_change_function: 0.1,
        p_change_input: 0.1,
        p_flip_input: 0.1,
        p_change_action: 0.1,
        p_delete_program: 0.5,
        p_add_program: 0.5,
        fitness_threshold: 0.0,
        legal_functions: vec![
            Function::Relu,
            Function::Plus,
            Function::Minus,
            Function::Times,
            Function::Divide,
            Function::Square,
            Function::Sin,
            Function::Log,
            Function::And,
            Function::Or,
            Function::Not,
            Function::Xor,
            Function::Min,
            Function::Max,
            Function::Greater,
            Function::Less,
            Function::IfThenElse,
            Function::Copy,
        ],
        constant_list: vec![0.0, 1.0, -1.0],
        feature_names: vec!["food-at-up", "food-at-down", "food-at-left", "food-at-right",
        "food-at-up-right", "food-at-down-right", "food-at-up-left", "food-at-down-left"]
    };

    for run in 1..=ant_trail_parameters.run_count {
        best_teams.push(crate::one_run(
            run,
            rng,
            &[],
            &[],
            &ant_trail_parameters,
            ant_trail_individual_error,
            index_to_ant_trail_action,
            &mut id_counter,
            dump,
            seed,
        ));
    }
    (best_teams, ant_trail_parameters)
}