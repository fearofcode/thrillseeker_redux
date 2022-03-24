use std::{env, fs, process, thread, time};
use thrillseeker_lib::{evaluate_team, Function, ProblemParameters};
use thrillseeker_lib::ant_trail::{Direction, Grid, MAXIMUM_MOVEMENTS, SANTA_FE_PERFECT_SCORE, WorldPosition};
use thrillseeker_lib::ant_trail_problem::AntTrailAction;
use thrillseeker_lib::Team;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Need to supply a JSON file to read in");
        process::exit(1);
    }

    let path = args[1].clone();

    let model_contents = fs::read_to_string(path).expect("Could not read file");

    let team: Team<AntTrailAction> = serde_json::from_str(&model_contents).unwrap();

    let params = ProblemParameters {
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

    let mut grid = Grid::santa_fe_trail();

    let mut food_gathered = 0;

    let mut pos = WorldPosition::new();

    if grid.food_at_position(pos) {
        food_gathered += 1;
        grid.remove_food_at_position(pos);
    }

    let mut movement_count = 0;

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
        let outputs = crate::evaluate_team(&team, &[state], &params);
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
                println!("Perfect score attained");
            }
            grid.remove_food_at_position(pos);
        }
        movement_count += 1;
        println!("Movement {}:", movement_count);
        grid.draw_with_position(pos);
        println!();

        thread::sleep(time::Duration::from_millis(200));
    }

    println!("Food gathered: {}. Movement count: {}", food_gathered, movement_count);
}