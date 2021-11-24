use crate::{Function, ProblemParameters, Team};
use fastrand::Rng;
use std::fmt;
use std::path::Path;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum WineClassification {
    // Only scores 3-9 appear in the dataset. No point forcing the code to learn data that doesn't
    // exist
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}

pub fn index_to_wine_action(index: usize) -> WineClassification {
    match index {
        0 => WineClassification::Three,
        1 => WineClassification::Four,
        2 => WineClassification::Five,
        3 => WineClassification::Six,
        4 => WineClassification::Seven,
        5 => WineClassification::Eight,
        6 => WineClassification::Nine,
        _ => panic!(),
    }
}

fn data_to_wine_action(index: usize) -> WineClassification {
    match index {
        3 => WineClassification::Three,
        4 => WineClassification::Four,
        5 => WineClassification::Five,
        6 => WineClassification::Six,
        7 => WineClassification::Seven,
        8 => WineClassification::Eight,
        9 => WineClassification::Nine,
        _ => panic!(),
    }
}
fn wine_action_numerical_value(action: &WineClassification) -> f32 {
    match action {
        WineClassification::Three => 3.0,
        WineClassification::Four => 4.0,
        WineClassification::Five => 5.0,
        WineClassification::Six => 6.0,
        WineClassification::Seven => 7.0,
        WineClassification::Eight => 8.0,
        WineClassification::Nine => 9.0,
    }
}
impl fmt::Display for WineClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WineClassification::Three => write!(f, "3"),
            WineClassification::Four => write!(f, "4"),
            WineClassification::Five => write!(f, "5"),
            WineClassification::Six => write!(f, "6"),
            WineClassification::Seven => write!(f, "7"),
            WineClassification::Eight => write!(f, "8"),
            WineClassification::Nine => write!(f, "9"),
        }
    }
}

pub fn wine_mean_absolute_deviance(
    team: &Team<WineClassification>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters,
    labels: &[WineClassification],
) -> (f32, Vec<WineClassification>) {
    let state = fitness_cases.to_owned();

    let outputs = crate::evaluate_team(team, &state, params);

    let deviation_sum: f32 = outputs
        .iter()
        .zip(labels.iter())
        .map(|(predicted, actual)| {
            (wine_action_numerical_value(predicted) - wine_action_numerical_value(actual)).abs()
        })
        .sum();

    (deviation_sum / (fitness_cases.len() as f32), outputs)
}

pub fn wine_runs(seed: u64, dump: bool, mut rng: &mut Rng) -> Vec<Team<WineClassification>> {
    let mut id_counter: u64 = 1;

    let mut best_teams: Vec<Team<WineClassification>> = vec![];

    let mut fitness_cases: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<WineClassification> = vec![];

    let wine_params = ProblemParameters {
        input_count: 11,
        register_count: 4,
        population_size: 5000,
        population_to_delete: 3000,
        approximate_fitness_case_count: 1599 + 4898,
        max_program_size: 64,
        min_initial_program_size: 1,
        max_initial_program_size: 16,
        // 3, 4, 5, 6, 7, 8, 9
        action_count: 7,
        max_initial_team_size: 6,
        max_team_size: 15,
        generation_count: 1000,
        generation_stagnation_limit: 10,
        run_count: 5,
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
        // better than SVM
        fitness_threshold: 0.45,
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
            // Function::Copy,
        ],
        constant_list: vec![
            0.0, 1.0,  2.0,  3.0,  4.0,  5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
             0.1,  0.2,  0.3,  0.4,  0.5,
            -0.1, -0.2, -0.3, -0.4, -0.5,
        ],
    };

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(Path::new("data/winequality-red.csv"))
        .unwrap();
    for result in rdr.records() {
        let record = result.unwrap();
        let numbers: Vec<f32> = record.iter().map(|s| s.parse().unwrap()).collect();
        let input = numbers[0..10].to_owned();
        let label = numbers[11] as usize;
        fitness_cases.push(crate::fitness_case_with_constants(input, &wine_params));
        labels.push(data_to_wine_action(label));
    }

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        .from_path(Path::new("data/winequality-white.csv"))
        .unwrap();
    for result in rdr.records() {
        let record = result.unwrap();
        let numbers: Vec<f32> = record.iter().map(|s| s.parse().unwrap()).collect();
        let input = numbers[0..10].to_owned();
        let label = numbers[11] as usize;
        fitness_cases.push(crate::fitness_case_with_constants(input, &wine_params));
        labels.push(data_to_wine_action(label));
    }

    for run in 1..=wine_params.run_count {
        best_teams.push(crate::one_run(
            run,
            &mut rng,
            &fitness_cases,
            &labels,
            &wine_params,
            wine_mean_absolute_deviance,
            index_to_wine_action,
            &mut id_counter,
            dump,
            seed,
        ));
    }
    best_teams
}
