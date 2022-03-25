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

    // TODO reuse extracted code from ant_trail_problem. don't copy and paste. it changes too
    // often and is counter productive for the casual hacking we're doing

    println!("Food gathered: {}. Movement count: {}", food_gathered, movement_count);
}