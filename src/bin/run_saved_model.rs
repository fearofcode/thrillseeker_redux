use std::time::Duration;
use std::{env, fs, process, thread};
use thrillseeker_lib::ant_trail::{Grid, WorldPosition};
use thrillseeker_lib::ant_trail_problem::{
    ant_trail_parameters, simulate_ant_trail, AntTrailAction, AntTrailFitness,
};
use thrillseeker_lib::Team;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Need to supply a JSON file to read in");
        process::exit(1);
    }

    let path = args[1].clone();

    let model_contents = fs::read_to_string(path).expect("Could not read file");

    let team: Team<AntTrailAction, AntTrailFitness> =
        serde_json::from_str(&model_contents).unwrap();

    let params = ant_trail_parameters();

    let draw_grid: fn(Grid, WorldPosition) = |grid, pos| {
        grid.draw_with_position(pos);
        thread::sleep(Duration::from_millis(100));
        println!();
    };

    let (food_gathered, steps, _) = simulate_ant_trail(&team, &params, Some(draw_grid));

    println!("Food gathered: {}", food_gathered);
    println!("Steps: {}", steps);
}
