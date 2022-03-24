use std::{env, fs, process};
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

    println!("{}", team);
}