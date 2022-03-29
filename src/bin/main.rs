use clap::{App, Arg};
use fastrand::Rng;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::hash::{Hash};
use std::io::prelude::*;
use thrillseeker_lib::{get_seed_value, ProblemParameters, Team};
use thrillseeker_lib::ant_trail::Grid;
use thrillseeker_lib::ant_trail_problem::ant_trail_runs;

fn setup() -> (u64, bool, Rng) {
    let matches = App::new("thrillseeker")
        .arg(
            Arg::with_name("seed")
                .short("s")
                .long("seed")
                .value_name("seed")
                .help("Seed the program with a given value"),
        )
        .arg(
            Arg::with_name("dump")
                .short("d")
                .long("dump")
                .help("Dump programs to a text file in the `dump` directory"),
        )
        .get_matches();

    let seed = if let Some(seed_arg) = matches.value_of("seed") {
        seed_arg.parse().unwrap()
    } else {
        get_seed_value()
    };

    let dump = matches.is_present("dump");

    println!("Using seed value {}. Dump = {}", seed, dump);

    let rng = Rng::with_seed(seed);
    (seed, dump, rng)
}

fn print_best_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(best_teams: &[Team<A>], params: &ProblemParameters) {
    println!("Best teams:");

    let constant_strings: Vec<String> = params.constant_list.iter().map(|c| c.to_string())
        .collect();
    let constant_strs: Vec<&str> = constant_strings.iter().map(|c| c.as_str()).collect();

    for best_team in best_teams.iter() {
        println!("Fitness {}:", best_team.fitness.unwrap());
        best_team.print_readable_features(&params.feature_names, &constant_strs);
    }
}

fn main() {
    let (seed, dump, mut rng) = setup();

    // let (best_teams, params) = acrobot::acrobot_runs(seed, dump, &mut rng);
    // print_best_teams(best_teams, &params);

    let (best_teams, params) = ant_trail_runs(seed, dump, &mut rng);
    print_best_teams(&best_teams, &params);

    // we can always run this since it's not much data
    fs::create_dir_all(format!("champions/{}", seed)).unwrap();

    for (run_idx, best_team) in best_teams.iter().enumerate() {
        let output_path = format!("champions/{}/run{}.json", seed, run_idx+1);
        let serialized = serde_json::to_string(best_team).unwrap();

        let mut file = File::create(output_path.clone()).unwrap();

        write!(file, "{}", serialized).unwrap();

        println!("Wrote champion for run {} to {}", run_idx+1, output_path);
    }

    println!("Ran with seed {}", seed);
}
