use crate::{Function, ProblemParameters, Team};
use fastrand::Rng;
use std::fmt;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum AcrobotAction {
    NegativeTorque,
    DoNothing,
    PositiveTorque,
}

pub fn index_to_acrobot_action(index: usize) -> AcrobotAction {
    match index {
        0 => AcrobotAction::NegativeTorque,
        1 => AcrobotAction::DoNothing,
        2 => AcrobotAction::PositiveTorque,
        _ => panic!(),
    }
}

impl fmt::Display for AcrobotAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcrobotAction::NegativeTorque => write!(f, "Negative_Torque"),
            AcrobotAction::DoNothing => write!(f, "Do_Nothing"),
            AcrobotAction::PositiveTorque => write!(f, "Positive_Torque"),
        }
    }
}

fn wrap(x: f32, m1: f32, m2: f32) -> f32 {
    let mut x2 = x;
    let diff = m2 - m1;
    while x2 > m2 {
        x2 -= diff;
    }

    while x2 < m1 {
        x2 += diff;
    }

    x2
}

fn bound(x: f32, m1: f32, m2: f32) -> f32 {
    x.max(m1).min(m2)
}

const NEGATIVE_TORQUE: f32 = -1.0;
const POSITIVE_TORQUE: f32 = 1.0;

const DT: f32 = 0.2;

const LINK_LENGTH_1: f32 = 1.0;
const LINK_MASS_1: f32 = 1.0;
const LINK_MASS_2: f32 = 1.0;
const LINK_COM_POS_1: f32 = 0.5;
const LINK_COM_POS_2: f32 = 0.5;
const LINK_MOI: f32 = 1.0;

const M_PI: f32 = std::f32::consts::PI;
const MAX_VEL_1: f32 = 4.0 * M_PI;
const MAX_VEL_2: f32 = 9.0 * M_PI;

const M1: f32 = LINK_MASS_1;
const M2: f32 = LINK_MASS_2;
const L1: f32 = LINK_LENGTH_1;
const LC1: f32 = LINK_COM_POS_1;
const LC2: f32 = LINK_COM_POS_2;
const I1: f32 = LINK_MOI;
const I2: f32 = LINK_MOI;
const G: f32 = 9.8;
const PI: f32 = M_PI;

fn square(x: f32) -> f32 {
    x * x
}

fn state_derivative(augmented_state: [f32; 5]) -> [f32; 5] {
    let a = augmented_state[4];
    let theta1 = augmented_state[0];
    let theta2 = augmented_state[1];
    let dtheta1 = augmented_state[2];
    let dtheta2 = augmented_state[3];

    let theta2_cos = theta2.cos();
    let sin_theta2 = theta2.sin();
    let d1 =
        M1 * square(LC1) + M2 * (square(L1) + square(LC2) + 2.0 * L1 * LC2 * theta2_cos) + I1 + I2;
    let d2 = M2 * (square(LC2) + L1 * LC2 * theta2_cos) + I2;
    let phi2 = M2 * LC2 * G * (theta1 + theta2 - PI / 2.0).cos();
    let phi1 = -M2 * L1 * LC2 * square(dtheta2) * sin_theta2
        - 2.0 * M2 * L1 * LC2 * dtheta2 * dtheta1 * sin_theta2
        + (M1 * LC1 + M2 * L1) * G * (theta1 - PI / 2.0).cos()
        + phi2;

    let ddtheta2 = (a + d2 / d1 * phi1 - M2 * L1 * LC2 * square(dtheta1) * sin_theta2 - phi2)
        / (M2 * square(LC2) + I2 - square(d2) / d1);
    let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

    [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.]
}

fn a_plus_by(a: [f32; 5], b: f32, y: [f32; 5]) -> [f32; 5] {
    [
        a[0] + b * y[0],
        a[1] + b * y[1],
        a[2] + b * y[2],
        a[3] + b * y[3],
        a[4] + b * y[4],
    ]
}

fn vector_sum(x: [f32; 5], y: [f32; 5]) -> [f32; 5] {
    [
        x[0] + y[0],
        x[1] + y[1],
        x[2] + y[2],
        x[3] + y[3],
        x[4] + y[4],
    ]
}

fn runge_kutta(y0: [f32; 5], t: [f32; 2]) -> [[f32; 5]; 2] {
    let mut y_out = [[0.0; 5]; 2];
    y_out[0] = y0;

    let i = 0;

    let thist = t[i];
    let dt = t[i + 1] - thist;

    let dt2 = dt / 2.0;

    let k1 = state_derivative(y0);
    let k2 = state_derivative(a_plus_by(y0, dt2, k1));
    let k3 = state_derivative(a_plus_by(y0, dt2, k2));
    let k4 = state_derivative(a_plus_by(y0, dt, k3));
    //   Y_Out[I + 1] = Y0 + (dt / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4);
    y_out[i + 1] = a_plus_by(
        y0,
        dt / 6.0,
        vector_sum(a_plus_by(k1, 2.0, k2), a_plus_by(k4, 2.0, k3)),
    );

    y_out
}

pub fn acrobot_individual_error(
    team: &Team<AcrobotAction>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters,
    _unused_labels: &[AcrobotAction],
) -> f32 {
    let mut steps: Vec<usize> = vec![0; fitness_cases.len()];

    let mut total_steps = 0;

    let mut state = fitness_cases.to_owned();

    let episode_limit = 500;

    loop {
        if state.is_empty() {
            break;
        }

        let outputs = crate::evaluate_team(team, &state, params);

        let mut to_delete = vec![false; outputs.len()];

        for (output_index, (current_state, output)) in state.iter_mut().zip(&outputs).enumerate() {
            let torque = match output {
                AcrobotAction::NegativeTorque => NEGATIVE_TORQUE,
                AcrobotAction::DoNothing => 0.0,
                AcrobotAction::PositiveTorque => POSITIVE_TORQUE,
            };

            let augmented_state = [
                current_state[0],
                current_state[1],
                current_state[2],
                current_state[3],
                torque,
            ];

            let integrated = runge_kutta(augmented_state, [0.0, DT]);
            let new_state = integrated[1];

            current_state[0] = wrap(new_state[0], -PI, PI);
            current_state[1] = wrap(new_state[1], -PI, PI);
            current_state[2] = bound(new_state[2], -MAX_VEL_1, MAX_VEL_1);
            current_state[3] = bound(new_state[3], -MAX_VEL_2, MAX_VEL_2);

            let done =
                (-current_state[0].cos() - (current_state[1] + current_state[0]).cos()) > 1.0;
            if done || (steps[output_index] >= episode_limit) {
                total_steps += steps[output_index];
                to_delete[output_index] = true;
            } else {
                steps[output_index] += 1;
            }
        }

        state = state
            .into_iter()
            .zip(to_delete)
            .filter(|(_, delete)| !(*delete))
            .map(|(s, _)| s)
            .collect();
    }

    total_steps as f32
}

pub fn acrobot_runs(seed: u64, dump: bool, mut rng: &mut Rng) -> (Vec<Team<AcrobotAction>>, ProblemParameters) {
    let mut id_counter: u64 = 1;

    let mut best_teams: Vec<Team<AcrobotAction>> = vec![];

    let mut fitness_cases: Vec<Vec<f32>> = vec![];

    let acrobot_parameters = ProblemParameters {
        input_count: 4,
        register_count: 4,
        population_size: 5000,
        population_to_delete: 4500,
        max_program_size: 32,
        min_initial_program_size: 1,
        max_initial_program_size: 8,
        action_count: 3,
        max_initial_team_size: 6,
        max_team_size: 15,
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
        fitness_threshold: 45.0 * (100.0) + 1.0,
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
        constant_list: vec![0.0],
        feature_names: vec!["x1", "v1", "x2", "v2"]
    };

    for _ in 0..100 {
        let x1 = crate::random_float_in_range(rng, -0.1, 0.1);
        let v1 = crate::random_float_in_range(rng, -0.1, 0.1);
        let x2 = crate::random_float_in_range(rng, -0.1, 0.1);
        let v2 = crate::random_float_in_range(rng, -0.1, 0.1);

        fitness_cases.push(crate::fitness_case_with_constants(
            vec![x1, v1, x2, v2],
            &acrobot_parameters,
        ));
    }

    for run in 1..=acrobot_parameters.run_count {
        best_teams.push(crate::one_run(
            run,
            rng,
            &fitness_cases,
            &[],
            &acrobot_parameters,
            acrobot_individual_error,
            index_to_acrobot_action,
            &mut id_counter,
            dump,
            seed,
        ));
    }
    (best_teams, acrobot_parameters)
}
