pub mod acrobot;
pub mod ant_trail;
pub mod ant_trail_problem;
mod lsh;

use fastrand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn coin_flip(p: f32, rng: &mut Rng) -> bool {
    rng.f32() < p
}

fn random_float_in_range(rng: &mut Rng, lower: f32, upper: f32) -> f32 {
    lower + (upper - lower) * rng.f32()
}

// increment this when changes thta would invalidate serialized teams occur
const THRILLSEEKER_VERSION: usize = 1;

const EVALUATE_PARALLEL: bool = true;

// this can be treated as problem-independent even though a given problem might pick a subset where
// the actual effective max arity is lower. in general, this code views the function set as relatively
// problem independent so the slight waste in space in each instruction is something that can be dealt with later
const MAX_ARITY: usize = 3;

pub struct ProblemParameters<
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    pub input_count: usize,
    pub register_count: usize,
    pub population_size: usize,
    pub keep_by_fitness: usize,
    pub keep_by_novelty: usize,
    pub select_by_novelty: usize,
    pub max_program_size: usize,
    pub min_initial_program_size: usize,
    pub max_initial_program_size: usize,
    pub action_count: usize,
    pub max_initial_team_size: usize,
    pub max_team_size: usize,
    pub tournament_size: usize,
    pub generation_count: usize,
    pub generation_stagnation_limit: usize,
    pub run_count: usize,
    pub p_delete_instruction: f32,
    pub p_add_instruction: f32,
    pub p_swap_instructions: f32,
    pub p_change_destination: f32,
    pub p_change_function: f32,
    pub p_change_input: f32,
    pub p_flip_input: f32,
    pub p_change_action: f32,
    pub p_delete_program: f32,
    pub p_add_program: f32,
    pub fitness_threshold: Fitness,
    pub legal_functions: Vec<Function>,
    pub constant_list: Vec<f32>,
    pub feature_names: Vec<&'static str>,
}

impl<
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > ProblemParameters<Fitness>
{
    fn deletion_point(&self) -> usize {
        self.keep_by_novelty + self.keep_by_fitness
    }

    fn fitness_case_size(&self) -> usize {
        self.input_count + self.constant_list.len()
    }
}

#[derive(Serialize, Deserialize, Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
pub enum Function {
    Relu,
    Plus,
    Minus,
    Times,
    Divide,
    Square,
    Sin,
    Log,
    And,
    Or,
    Not,
    Xor,
    Min,
    Max,
    Greater,
    Less,
    IfThenElse,
    Copy,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Function::Relu => write!(f, "relu"),
            Function::Plus => write!(f, "+"),
            Function::Minus => write!(f, "-"),
            Function::Times => write!(f, "*"),
            Function::Divide => write!(f, "/"),
            Function::Square => write!(f, "sq"),
            Function::Sin => write!(f, "sin"),
            Function::Log => write!(f, "log"),
            Function::And => write!(f, "&&"),
            Function::Or => write!(f, "||"),
            Function::Not => write!(f, "!"),
            Function::Xor => write!(f, "^"),
            Function::Min => write!(f, "min"),
            Function::Max => write!(f, "max"),
            Function::Greater => write!(f, ">"),
            Function::Less => write!(f, "<"),
            Function::IfThenElse => write!(f, "if-then-else"),
            Function::Copy => write!(f, "copy"),
        }
    }
}

fn function_arity(f: &Function) -> usize {
    match f {
        Function::Relu => 1,
        Function::Plus => 2,
        Function::Minus => 2,
        Function::Times => 2,
        Function::Divide => 2,
        Function::Square => 1,
        Function::Sin => 1,
        Function::Log => 1,
        Function::And => 2,
        Function::Or => 2,
        Function::Not => 1,
        Function::Xor => 2,
        Function::Min => 2,
        Function::Max => 2,
        Function::Greater => 2,
        Function::Less => 2,
        Function::IfThenElse => 3,
        Function::Copy => 1,
    }
}

fn infix_op(f: &Function) -> bool {
    match f {
        Function::Relu => false,
        Function::Plus => true,
        Function::Minus => true,
        Function::Times => true,
        Function::Divide => true,
        Function::Square => false,
        Function::Sin => false,
        Function::Log => false,
        Function::And => true,
        Function::Or => true,
        Function::Not => true,
        Function::Xor => true,
        Function::Min => false,
        Function::Max => false,
        Function::Greater => true,
        Function::Less => true,
        Function::IfThenElse => false,
        Function::Copy => false,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
struct Instruction {
    destination: usize,
    operands: [usize; MAX_ARITY],
    is_register: [bool; MAX_ARITY],
    op: Function,
    index: usize,
}

impl Instruction {
    fn print_readable_features(&self, feature_names: &[&'static str], constants: &[&str]) {
        print!("\t\tr[{}] = ", self.destination);

        let arity = function_arity(&self.op);
        let is_register0 = self.is_register[0];
        let register0_string = format!("r[{}]", self.operands[0]);
        let register1_string = format!("r[{}]", self.operands[1]);

        let register0_str = if is_register0 {
            register0_string.as_str()
        } else {
            let op = self.operands[0];
            if op < feature_names.len() {
                feature_names[op]
            } else {
                constants[op - feature_names.len()]
            }
        };
        let is_register1 = self.is_register[1];
        let register1_str = if is_register1 {
            register1_string.as_str()
        } else {
            let op = self.operands[1];
            if op < feature_names.len() {
                feature_names[op]
            } else {
                constants[op - feature_names.len()]
            }
        };

        if arity == 1 {
            print!("{}", self.op);
            if !infix_op(&self.op) {
                print!("(");
            }
            print!("{}", register0_str);
            if !infix_op(&self.op) {
                println!(");");
            } else {
                println!(";");
            }
        } else if arity == 2 {
            if infix_op(&self.op) {
                println!("{} {} {};", register0_str, self.op, register1_str)
            } else {
                println!("{}({}, {});", self.op, register0_str, register1_str);
            }
        } else {
            print!("{}(", self.op);
            for i in 0..arity {
                if self.is_register[i] {
                    print!("r[{}]", self.operands[i]);
                } else {
                    let register_str = {
                        let op = self.operands[i];
                        if op < feature_names.len() {
                            feature_names[op]
                        } else {
                            constants[op - feature_names.len()]
                        }
                    };
                    print!("{}", register_str);
                }
                if i != (arity - 1) {
                    print!(", ");
                }
            }
            println!(");");
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\t\tr[{}] = ", self.destination).unwrap();

        let arity = function_arity(&self.op);
        let is_register0 = self.is_register[0];
        let register0_char = if is_register0 { 'r' } else { 'i' };
        let is_register1 = self.is_register[1];
        let register1_char = if is_register1 { 'r' } else { 'i' };

        if arity == 1 {
            write!(f, "{}", self.op).unwrap();
            if !infix_op(&self.op) {
                write!(f, "(").unwrap();
            }
            write!(f, "{}[{}]", register0_char, self.operands[0]).unwrap();
            if !infix_op(&self.op) {
                writeln!(f, ");").unwrap();
            } else {
                writeln!(f, ";").unwrap();
            }
        } else if arity == 2 {
            if infix_op(&self.op) {
                writeln!(
                    f,
                    "{}[{}] {} {}[{}];",
                    register0_char, self.operands[0], self.op, register1_char, self.operands[1]
                )
                .unwrap();
            } else {
                writeln!(
                    f,
                    "{}({}[{}], {}[{}]);",
                    self.op, register0_char, self.operands[0], register1_char, self.operands[1]
                )
                .unwrap();
            }
        } else {
            write!(f, "{}(", self.op).unwrap();
            for i in 0..arity {
                write!(
                    f,
                    "{}[{}]",
                    if self.is_register[i] { 'r' } else { 'i' },
                    self.operands[i]
                )
                .unwrap();

                if i != (arity - 1) {
                    write!(f, ", ").unwrap();
                }
            }
            writeln!(f, ");").unwrap();
        }
        Ok(())
    }
}

impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        (self.destination == other.destination)
            && (self.op == other.op)
            && (self.operands == other.operands)
            && (self.is_register == other.is_register)
    }
}

impl Eq for Instruction {}

impl Hash for Instruction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.destination);
        for operand in self.operands.iter() {
            state.write_usize(*operand);
        }
        for is_register in self.is_register.iter() {
            is_register.hash(state);
        }
        self.op.hash(state);
        state.finish();
    }
}

fn random_instruction<
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    rng: &mut Rng,
    params: &ProblemParameters<Fitness>,
) -> Instruction {
    let mut instruction = Instruction {
        destination: 0,
        operands: [0; MAX_ARITY],
        is_register: [false; MAX_ARITY],
        op: Function::Relu,
        index: 0,
    };

    instruction.destination = rng.usize(..params.register_count);

    let function_index = rng.usize(..params.legal_functions.len());
    let random_function = params.legal_functions[function_index];
    instruction.op = random_function;
    let arity = function_arity(&random_function);

    for i in 0..arity {
        let is_register = rng.bool();
        instruction.is_register[i] = is_register;

        let index_range = if is_register {
            params.register_count
        } else {
            params.fitness_case_size()
        };

        let input_index = rng.usize(..index_range);
        instruction.operands[i] = input_index;
    }

    instruction
}

fn active_instructions_from_index<
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    instructions: &[Instruction],
    starting_index: usize,
    params: &ProblemParameters<Fitness>,
) -> Vec<bool> {
    let mut active_instructions = vec![false; params.max_program_size];
    let starting_instruction = instructions[starting_index];

    let mut referenced_registers = HashSet::new();

    let arity = function_arity(&starting_instruction.op);

    for (operand_index, operand) in starting_instruction.operands.iter().take(arity).enumerate() {
        if starting_instruction.is_register[operand_index] {
            referenced_registers.insert(*operand);
        }
    }

    active_instructions[starting_index] = true;

    if referenced_registers.is_empty() || starting_index == 0 {
        return active_instructions;
    }

    for index in (0..=(starting_index - 1)).rev() {
        let previous_instruction = instructions[index];
        // an instruction is active if its destination is a register that is referenced by an active instruction
        if referenced_registers.contains(&previous_instruction.destination) {
            referenced_registers.remove(&previous_instruction.destination);

            let arity = function_arity(&previous_instruction.op);

            for (operand_index, operand) in
                previous_instruction.operands.iter().take(arity).enumerate()
            {
                if previous_instruction.is_register[operand_index] {
                    referenced_registers.insert(*operand);
                }
            }

            active_instructions[index] = true;
        }
    }

    active_instructions
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Program<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    active_instructions: Vec<Instruction>,
    introns: Vec<Instruction>,
    action: A,
    id: u64,
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Program<A>
{
    fn print_readable_features(&self, feature_names: &[&'static str], constants: &[&str]) {
        println!("\t\tAction: {}", self.action);
        println!("\t\tID #: {}", self.id);

        if self.active_instructions.is_empty() {
            println!("\t\t(Empty program)");
        } else {
            for instruction in self.active_instructions.iter() {
                instruction.print_readable_features(feature_names, constants);
            }
        }
    }

    fn restore_introns(&mut self) {
        for intron in self.introns.iter() {
            if intron.index > self.active_instructions.len() {
                self.active_instructions.push(*intron);
            } else {
                self.active_instructions.insert(intron.index, *intron);
            }
        }
        self.introns.clear();
    }

    fn mark_introns<
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    >(
        &mut self,
        params: &ProblemParameters<Fitness>,
    ) {
        self.restore_introns();

        // remark indexes
        for index in 0..self.active_instructions.len() {
            self.active_instructions[index].index = index;
        }

        // look for the first instruction with a destination of the output register

        let mut last_output_index = 0;
        let mut found_active_instruction = false;
        for (instruction_index, instruction) in self.active_instructions.iter().enumerate().rev() {
            // r[0] is assumed to be the output register
            if instruction.destination == 0 {
                last_output_index = instruction_index;
                found_active_instruction = true;
                break;
            } else {
                // end intron
                self.introns.push(*instruction);
            }
        }

        if !found_active_instruction {
            self.active_instructions.clear();
            return;
        }

        // avoid deleting from the vector while iterating through it
        for instruction_index in ((last_output_index + 1)..(self.active_instructions.len())).rev() {
            self.active_instructions.remove(instruction_index);
        }

        let active_instructions =
            active_instructions_from_index(&self.active_instructions, last_output_index, params);

        let mut new_active_instructions = vec![];
        new_active_instructions.reserve(params.max_program_size);

        for (active_instruction_index, active_instruction) in active_instructions
            .iter()
            .enumerate()
            .take(last_output_index + 1)
        {
            if !active_instruction {
                self.introns
                    .push(self.active_instructions[active_instruction_index]);
            } else {
                new_active_instructions.push(self.active_instructions[active_instruction_index]);
            }
        }

        self.active_instructions = new_active_instructions;

        self.introns.sort_by_key(|i| i.index);

        // remark indexes
        for index in 0..self.active_instructions.len() {
            self.active_instructions[index].index = index;
        }
    }
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Hash for Program<A>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        for instruction in self.active_instructions.iter() {
            instruction.hash(state);
        }
    }
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > PartialEq for Program<A>
{
    fn eq(&self, other: &Self) -> bool {
        self.action == other.action
            && self
                .active_instructions
                .iter()
                .zip(&other.active_instructions)
                .all(|(i1, i2)| i1 == i2)
    }
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Eq for Program<A>
{
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > fmt::Display for Program<A>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "\t\tAction: {}", self.action).unwrap();
        writeln!(f, "\t\tID #: {}", self.id).unwrap();

        if self.active_instructions.is_empty() {
            writeln!(f, "\t\t(Empty program)").unwrap();
        } else {
            for instruction in self.active_instructions.iter() {
                write!(f, "{}", instruction).unwrap();
            }
        }
        Ok(())
    }
}

fn evaluate_program<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    program: &Program<A>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters<Fitness>,
) -> Vec<f32> {
    let mut registers: Vec<Vec<f32>> = vec![vec![0.0; params.register_count]; fitness_cases.len()];

    for instruction in program.active_instructions.iter() {
        let destination = instruction.destination;
        let operands = instruction.operands;
        let operand1 = operands[0];
        let operand2 = operands[1];
        let operand1_is_register = instruction.is_register[0];
        let operand2_is_register = instruction.is_register[1];

        let instruction_op = instruction.op;

        match instruction_op {
            Function::Relu => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        register_set[destination] = op1.max(0.0);
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        register_set[destination] = op1.max(0.0);
                    }
                }
            }
            Function::Plus => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = op1 + op2;
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = op1 + op2;
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = op1 + op2;
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = op1 + op2;
                    }
                }
            }
            Function::Minus => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = op1 - op2;
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = op1 - op2;
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = op1 - op2;
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = op1 - op2;
                    }
                }
            }
            Function::Times => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = op1 * op2;
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = op1 * op2;
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = op1 * op2;
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = op1 * op2;
                    }
                }
            }
            Function::Divide => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            if op2 != 0.0 {
                                register_set[destination] = op1 / op2;
                            } else {
                                register_set[destination] = 1.0;
                            }
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            if op2 != 0.0 {
                                register_set[destination] = op1 / op2;
                            } else {
                                register_set[destination] = 1.0;
                            }
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        if op2 != 0.0 {
                            register_set[destination] = op1 / op2;
                        } else {
                            register_set[destination] = 1.0;
                        }
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        if op2 != 0.0 {
                            register_set[destination] = op1 / op2;
                        } else {
                            register_set[destination] = 1.0;
                        }
                    }
                }
            }
            Function::Square => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        register_set[destination] = op1 * op1;
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        register_set[destination] = op1 * op1;
                    }
                }
            }
            Function::Sin => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        register_set[destination] = op1.sin();
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        register_set[destination] = op1.sin();
                    }
                }
            }
            Function::Log => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        if op1 != 0.0 {
                            register_set[destination] = op1.abs().log10();
                        }
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        if op1 != 0.0 {
                            register_set[destination] = op1.abs().log10();
                        }
                    }
                }
            }
            Function::And => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] =
                                if (op1 > 0.0) && (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] =
                                if (op1 > 0.0) && (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] =
                            if (op1 > 0.0) && (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] =
                            if (op1 > 0.0) && (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::Or => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] =
                                if (op1 > 0.0) || (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] =
                                if (op1 > 0.0) || (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] =
                            if (op1 > 0.0) || (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] =
                            if (op1 > 0.0) || (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::Not => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        register_set[destination] = if op1 <= 0.0 { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        register_set[destination] = if op1 <= 0.0 { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::Xor => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] =
                                if (op1 > 0.0) ^ (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] =
                                if (op1 > 0.0) ^ (op2 > 0.0) { 1.0 } else { 0.0 };
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] =
                            if (op1 > 0.0) ^ (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] =
                            if (op1 > 0.0) ^ (op2 > 0.0) { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::Min => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = op1.min(op2);
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = op1.min(op2);
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = op1.min(op2);
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = op1.min(op2);
                    }
                }
            }
            Function::Max => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = op1.max(op2);
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = op1.max(op2);
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = op1.max(op2);
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = op1.max(op2);
                    }
                }
            }
            Function::Greater => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = if op1 > op2 { 1.0 } else { 0.0 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = if op1 > op2 { 1.0 } else { 0.0 };
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = if op1 > op2 { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = if op1 > op2 { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::Less => {
                if operand1_is_register {
                    if operand2_is_register {
                        for register_set in registers.iter_mut() {
                            let op1 = register_set[operand1];
                            let op2 = register_set[operand2];
                            register_set[destination] = if op1 < op2 { 1.0 } else { 0.0 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            register_set[destination] = if op1 < op2 { 1.0 } else { 0.0 };
                        }
                    }
                } else if operand2_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = register_set[operand2];
                        register_set[destination] = if op1 < op2 { 1.0 } else { 0.0 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        register_set[destination] = if op1 < op2 { 1.0 } else { 0.0 };
                    }
                }
            }
            Function::IfThenElse => {
                let operand3 = operands[2];
                let operand3_is_register = instruction.is_register[2];
                if operand1_is_register {
                    if operand2_is_register {
                        if operand3_is_register {
                            for register_set in registers.iter_mut() {
                                let op1 = register_set[operand1];
                                let op2 = register_set[operand2];
                                let op3 = register_set[operand3];
                                register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                            }
                        } else {
                            for (register_index, register_set) in registers.iter_mut().enumerate() {
                                let op1 = register_set[operand1];
                                let op2 = register_set[operand2];
                                let op3 = fitness_cases[register_index][operand3];
                                register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                            }
                        }
                    } else if operand3_is_register {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            let op3 = register_set[operand3];
                            register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = register_set[operand1];
                            let op2 = fitness_cases[register_index][operand2];
                            let op3 = fitness_cases[register_index][operand3];
                            register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                        }
                    }
                } else if operand2_is_register {
                    if operand3_is_register {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = fitness_cases[register_index][operand1];
                            let op2 = register_set[operand2];
                            let op3 = register_set[operand3];
                            register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                        }
                    } else {
                        for (register_index, register_set) in registers.iter_mut().enumerate() {
                            let op1 = fitness_cases[register_index][operand1];
                            let op2 = register_set[operand2];
                            let op3 = fitness_cases[register_index][operand3];
                            register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                        }
                    }
                } else if operand3_is_register {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        let op3 = register_set[operand3];
                        register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        let op2 = fitness_cases[register_index][operand2];
                        let op3 = fitness_cases[register_index][operand3];
                        register_set[destination] = if op1 > 0.0 { op2 } else { op3 };
                    }
                }
            }
            Function::Copy => {
                if operand1_is_register {
                    for register_set in registers.iter_mut() {
                        let op1 = register_set[operand1];
                        register_set[destination] = op1;
                    }
                } else {
                    for (register_index, register_set) in registers.iter_mut().enumerate() {
                        let op1 = fitness_cases[register_index][operand1];
                        register_set[destination] = op1;
                    }
                }
            }
        }
    }

    registers
        .iter()
        .map(|r| 1.0 / (1.0 + (-r[0]).exp()))
        .collect()
}

fn size_fair_dependent_instruction_crossover<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    parent1: &mut Program<A>,
    parent2: &mut Program<A>,
    rng: &mut Rng,
    params: &ProblemParameters<Fitness>,
) {
    if parent1.active_instructions.is_empty() || parent2.active_instructions.is_empty() {
        return;
    }

    // 1. pick a random instruction in Parent1.
    // 2. calculate its subtree size (number of dependent instructions).
    // 3. find the subtree in Parent 2 with the closest subtree size, settling ties through distance from Parent1's
    //    crossover index.
    // 4. copy Parent2's subtree over to Parent1.

    // this is inspired by research on size fair/homologous crossover operators. see Langdon's "Size Fair and Homologous
    // Tree Genetic Programming Crossovers" and Francone et al's "Homologous Crossover in Genetic Programming" for the
    // kind of ideas that inspired this.

    // 1. pick a random instruction in Parent1.
    let parent1_crossover_index = rng.usize(..parent1.active_instructions.len());

    // 2. calculate its subtree size (number of dependent instructions).
    let parent1_active_indexes = active_instructions_from_index(
        &parent1.active_instructions,
        parent1_crossover_index,
        params,
    );
    let parent1_subtree_indexes: Vec<usize> = parent1_active_indexes
        .iter()
        .enumerate()
        .filter(|(_, elt)| **elt)
        .map(|(index, _)| index)
        .collect();
    let parent1_active_instruction_count = parent1_subtree_indexes.len();

    let parent2_subtree_sizes = (0..parent2.active_instructions.len()).map(|index| {
        active_instructions_from_index(&parent2.active_instructions, index, params)
            .iter()
            .filter(|index2| **index2)
            .count()
    });

    let closest_subtree_index: usize = parent2_subtree_sizes
        .into_iter()
        .enumerate()
        .map(|(subtree_index, subtree_size)| {
            (
                subtree_index,
                if parent1_active_instruction_count > subtree_size {
                    parent1_active_instruction_count - subtree_size
                } else {
                    subtree_size - parent1_active_instruction_count
                },
            )
        })
        .min_by(|(a_index, a_difference), (b_index, b_difference)| {
            match a_difference.cmp(b_difference) {
                Ordering::Equal => {
                    let a_index_difference =
                        ((*a_index as i64) - (parent1_crossover_index as i64)).abs();
                    let b_index_difference =
                        ((*b_index as i64) - (parent1_crossover_index as i64)).abs();
                    a_index_difference.cmp(&b_index_difference)
                }
                other => other,
            }
        })
        .map(|(index, _)| index)
        .unwrap();

    // 4. copy Parent2's subtree over to Parent1.
    let parent2_active_indexes =
        active_instructions_from_index(&parent2.active_instructions, closest_subtree_index, params);
    let parent2_subtree_indexes: Vec<usize> = parent2_active_indexes
        .iter()
        .enumerate()
        .filter(|(_, elt)| **elt)
        .map(|(index, _)| index)
        .collect();

    // copy respective instructions from Parent2 subtree into Parent1.

    let mut parent1_subtree_index = (parent1_subtree_indexes.len() - 1) as i64;
    let mut parent2_subtree_index = (parent2_subtree_indexes.len() - 1) as i64;

    while parent1_subtree_index >= 0 && parent2_subtree_index >= 0 {
        let parent1_instruction_index = parent1_subtree_indexes[parent1_subtree_index as usize];
        let parent2_instruction_index = parent2_subtree_indexes[parent2_subtree_index as usize];
        parent1.active_instructions[parent1_instruction_index] =
            parent2.active_instructions[parent2_instruction_index];
        parent1_subtree_index -= 1;
        parent2_subtree_index -= 1;
    }
}

fn mutate_program<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    program: &mut Program<A>,
    team_actions: &[A],
    rng: &mut Rng,
    counter: &mut u64,
    params: &ProblemParameters<Fitness>,
    index_to_program_action: fn(usize) -> A,
) {
    if program.active_instructions.is_empty() {
        return;
    }

    // only run one mutation per program at a time.
    if program.active_instructions.len() > 1 && coin_flip(params.p_delete_instruction, rng) {
        let delete_index = rng.usize(..program.active_instructions.len());
        program.active_instructions.remove(delete_index);
    } else if (program.active_instructions.len() + program.introns.len()) < params.max_program_size // don't make program too big
        && coin_flip(params.p_add_instruction, rng)
    {
        let add_index = rng.usize(..=program.active_instructions.len());
        let instruction = random_instruction(rng, params);
        program.active_instructions.insert(add_index, instruction);
    } else if program.active_instructions.len() >= 2 && coin_flip(params.p_swap_instructions, rng) {
        let index1 = rng.usize(..program.active_instructions.len());
        let index2 = rng.usize(..program.active_instructions.len());

        program.active_instructions.swap(index1, index2);
    } else if coin_flip(params.p_change_destination, rng) {
        let index = rng.usize(..program.active_instructions.len());
        let destination = rng.usize(..params.register_count);
        program.active_instructions[index].destination = destination;
    } else if coin_flip(params.p_change_function, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());
        let current_op = program.active_instructions[instruction_index].op;
        let current_arity = function_arity(&current_op);

        let equal_arity_functions: Vec<_> = params
            .legal_functions
            .iter()
            .filter(|f| function_arity(f) == current_arity)
            .collect();

        let equal_arity_function_count = equal_arity_functions.len();

        let new_function_index = rng.usize(..equal_arity_function_count);
        let new_op = equal_arity_functions[new_function_index];
        program.active_instructions[instruction_index].op = *new_op;
    } else if coin_flip(params.p_flip_input, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());
        let instruction_op = program.active_instructions[instruction_index].op;
        let arity = function_arity(&instruction_op);
        let input_index = rng.usize(..arity);

        let is_register = rng.bool();
        program.active_instructions[instruction_index].is_register[input_index] = is_register;

        let current_operand = program.active_instructions[instruction_index].operands[input_index];

        if current_operand >= params.register_count && is_register {
            program.active_instructions[instruction_index].operands[input_index] =
                rng.usize(..params.register_count);
        }
    } else if coin_flip(params.p_change_input, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());

        let instruction_op = program.active_instructions[instruction_index].op;
        let arity = function_arity(&instruction_op);
        let input_index = rng.usize(..arity);

        let is_register = program.active_instructions[instruction_index].is_register[input_index];

        let limit = if is_register {
            params.register_count
        } else {
            params.fitness_case_size()
        };

        program.active_instructions[instruction_index].operands[input_index] = rng.usize(..limit);
    } else if coin_flip(params.p_change_action, rng) {
        let action_index = rng.usize(..params.action_count);
        let new_action = index_to_program_action(action_index);

        let learners_with_action = team_actions
            .iter()
            .enumerate()
            .filter(|(program_index, action)| {
                *program_index != action_index && **action == new_action
            })
            .count();

        // only change action if there is another learner with this action so actions are not lost
        // if it is not beneficial to ever perform a certain action, learners will have to evolve
        // a no-op program
        if learners_with_action >= 1 {
            program.action = new_action;
        }
    }

    // effective instructions may have changed
    // program.mark_introns();
    *counter += 1;
    program.id = *counter;
}

type BehaviorDescriptor = String;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    programs: Vec<Program<A>>,
    pub fitness: Option<Fitness>,
    behavior_descriptor: Option<BehaviorDescriptor>,
    novelty: Option<f32>,
    id: u64,
    parent1_id: u64,
    parent2_id: u64,
    version: usize,
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Team<A, Fitness>
{
    pub fn print_readable_features(&self, feature_names: &[&'static str], constants: &[&str]) {
        println!("Team ID #{}", self.id);

        if self.parent1_id != 0 {
            println!("Team Parent #1: {}", self.parent1_id);
        } else {
            println!("Team Parent #1: None");
        }

        if self.parent2_id != 0 {
            println!("Team Parent #2: {}", self.parent2_id);
        } else {
            println!("Team Parent #2: None");
        }

        for (team_index, program) in self.programs.iter().enumerate() {
            println!("\tProgram #{}", team_index + 1);
            (*program).print_readable_features(feature_names, constants);
        }

        println!();
    }

    fn restore_introns(&mut self) {
        for program in self.programs.iter_mut() {
            program.restore_introns();
        }
    }

    fn mark_introns(&mut self, params: &ProblemParameters<Fitness>) {
        for program in self.programs.iter_mut() {
            program.mark_introns(params);
        }
    }

    fn active_instruction_count(&self) -> usize {
        let mut sum = 0;
        for program in self.programs.iter() {
            sum += program.active_instructions.len();
        }

        sum
    }
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > PartialEq for Team<A, Fitness>
{
    fn eq(&self, other: &Self) -> bool {
        if self.programs.len() != other.programs.len() {
            return false;
        }

        self.programs
            .iter()
            .zip(&other.programs)
            .all(|(p1, p2)| p1 == p2)
    }
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Eq for Team<A, Fitness>
{
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
        Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > fmt::Display for Team<A, Fitness>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Team ID #{}", self.id).unwrap();

        if self.parent1_id != 0 {
            writeln!(f, "Team Parent #1: {}", self.parent1_id).unwrap();
        } else {
            writeln!(f, "Team Parent #1: None").unwrap();
        }

        if self.parent2_id != 0 {
            writeln!(f, "Team Parent #2: {}", self.parent2_id).unwrap();
        } else {
            writeln!(f, "Team Parent #2: None").unwrap();
        }

        for (team_index, program) in self.programs.iter().enumerate() {
            writeln!(f, "\tProgram #{}", team_index + 1).unwrap();
            writeln!(f, "{}", *program).unwrap();
        }

        writeln!(f)
    }
}

fn initialize_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    rng: &mut Rng,
    id_counter: &mut u64,
    params: &ProblemParameters<Fitness>,
    index_to_program_action: fn(usize) -> A,
) -> Vec<Team<A, Fitness>> {
    let mut teams = vec![];
    teams.reserve(params.population_size);

    while teams.len() < params.population_size {
        let mut team = Team {
            programs: vec![],
            fitness: None,
            novelty: None,
            behavior_descriptor: None,
            id: 0,
            parent1_id: 0,
            parent2_id: 0,
            version: THRILLSEEKER_VERSION,
        };
        team.programs.reserve(params.max_team_size);

        let program_count = rng.usize(..params.max_initial_team_size);

        let mut program_set = HashSet::new();
        program_set.reserve(params.max_team_size);

        // make sure each team has at least one of each action type
        for action_index in 0..params.action_count {
            loop {
                let mut program = Program {
                    active_instructions: vec![],
                    introns: vec![],
                    action: index_to_program_action(action_index),
                    id: 0,
                };
                program.active_instructions.reserve(params.max_program_size);
                program.introns.reserve(params.max_program_size);

                let instruction_count =
                    rng.usize(params.min_initial_program_size..params.max_initial_program_size);

                for _ in 0..instruction_count {
                    let instruction = random_instruction(rng, params);
                    program.active_instructions.push(instruction);
                }

                program.mark_introns(params);

                // clear introns so that initially all code is active
                program.introns.clear();
                // just to set indexes properly
                program.mark_introns(params);

                if !program_set.contains(&program) && !program.active_instructions.is_empty() {
                    *id_counter += 1;
                    program.id = *id_counter;
                    team.programs.push(program.clone());
                    program_set.insert(program.clone());

                    break;
                }
            }
        }

        while team.programs.len() < program_count {
            let action = index_to_program_action(rng.usize(..params.action_count));
            loop {
                let mut program = Program {
                    active_instructions: vec![],
                    introns: vec![],
                    action,
                    id: 0,
                };
                program.active_instructions.reserve(params.max_program_size);
                program.introns.reserve(params.max_program_size);

                let instruction_count =
                    rng.usize(params.min_initial_program_size..params.max_initial_program_size);

                for _ in 0..instruction_count {
                    let instruction = random_instruction(rng, params);
                    program.active_instructions.push(instruction);
                }

                if !program_set.contains(&program) && !program.active_instructions.is_empty() {
                    *id_counter += 1;
                    program.id = *id_counter;
                    team.programs.push(program.clone());
                    program_set.insert(program.clone());

                    break;
                }
            }
        }

        *id_counter += 1;
        team.id = *id_counter;
        teams.push(team);
    }

    teams
}

pub fn evaluate_team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    team: &Team<A, Fitness>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters<Fitness>,
) -> Vec<A> {
    let team_outputs: Vec<Vec<f32>> = team
        .programs
        .iter()
        .map(|p| evaluate_program(p, fitness_cases, params))
        .collect();

    fitness_cases
        .iter()
        .enumerate()
        .map(|(fitness_case_index, _)| {
            team_outputs
                .iter()
                .enumerate()
                .map(|(index, output)| (index, output[fitness_case_index]))
                .filter(|(_, bid)| !bid.is_nan())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(index, _)| team.programs[index].action)
                .unwrap()
        })
        .collect()
}

#[derive(Debug, Eq, PartialEq)]
struct ArchiveEntry<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    team: Team<A, Fitness>,
    generation_added: usize,
}

#[derive(Debug)]
struct Archive<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    entries: HashMap<usize, ArchiveEntry<A, Fitness>>,
    distance_cache: HashMap<(usize, usize), f32>,
}

fn tournament_selection_fitness<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    teams: &[Team<A, Fitness>],
    rng: &mut Rng,
    params: &ProblemParameters<Fitness>,
) -> usize {
    (0..params.tournament_size)
        .map(|_| rng.usize(..params.deletion_point()))
        .max_by(|index1, index2| {
            match teams[*index1]
                .fitness
                .unwrap()
                .partial_cmp(&teams[*index2].fitness.unwrap())
                .unwrap()
            {
                Ordering::Equal => teams[*index1]
                    .active_instruction_count()
                    .partial_cmp(&teams[*index2].active_instruction_count())
                    .unwrap(),
                other => other,
            }
        })
        .unwrap()
}

fn tournament_selection_novelty<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    teams: &[Team<A, Fitness>],
    rng: &mut Rng,
    params: &ProblemParameters<Fitness>,
) -> usize {
    (0..params.tournament_size)
        .map(|_| rng.usize(..params.deletion_point()))
        .max_by(|index1, index2| {
            match teams[*index1]
                .novelty
                .unwrap()
                .partial_cmp(&teams[*index2].novelty.unwrap())
                .unwrap()
            {
                Ordering::Equal => teams[*index1]
                    .active_instruction_count()
                    .partial_cmp(&teams[*index2].active_instruction_count())
                    .unwrap(),
                other => other,
            }
        })
        .unwrap()
}

const MAX_MUTATION_CROSSOVER_ATTEMPTS: usize = 5;

fn mutate_team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    team: &mut Team<A, Fitness>,
    other_team: &mut Team<A, Fitness>,
    rng: &mut Rng,
    counter: &mut u64,
    params: &ProblemParameters<Fitness>,
    index_to_program_action: fn(usize) -> A,
) {
    let team_actions: Vec<_> = team.programs.iter().map(|p| p.action).collect();

    // prevent mutations that create empty programs
    team.programs = team
        .programs
        .iter_mut()
        .map(|program| {
            let original_program = program.clone();
            let mut mutated_program = original_program.clone();

            let mut retry_count = 0;

            loop {
                mutate_program(
                    &mut mutated_program,
                    &team_actions,
                    rng,
                    counter,
                    params,
                    index_to_program_action,
                );
                mutated_program.mark_introns(params);
                if !mutated_program.active_instructions.is_empty() {
                    break;
                }
                mutated_program = original_program.clone();
                retry_count += 1;
                // if a program can't possibly be mutated without creating an empty program, give up
                if retry_count > MAX_MUTATION_CROSSOVER_ATTEMPTS {
                    break;
                }
            }
            mutated_program
        })
        .collect();

    if team.programs.len() < params.max_team_size && coin_flip(params.p_add_program, rng) {
        let other_team_index = rng.usize(..other_team.programs.len());
        team.programs
            .push(other_team.programs[other_team_index].clone());
    }

    if team.programs.len() > 1 && coin_flip(params.p_delete_program, rng) {
        let deleted_index = rng.usize(..team.programs.len());
        let deleted_action = team.programs[deleted_index].action;

        let learners_with_action = team_actions
            .iter()
            .enumerate()
            .filter(|(program_index, action)| {
                *program_index != deleted_index && **action == deleted_action
            })
            .count();

        // only delete program if there is another learner with this action so actions are not lost
        // if it is not beneficial to ever perform a certain action, learners will have to evolve
        // a no-op program
        if learners_with_action >= 1 {
            team.programs.remove(deleted_index);
        }
    }
}

fn team_crossover<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    team1: &mut Team<A, Fitness>,
    team2: &mut Team<A, Fitness>,
    rng: &mut Rng,
    params: &ProblemParameters<Fitness>,
) {
    assert!(team1.programs.len() <= team2.programs.len());

    // shuffle programs so that we can iterate in order without bias
    rng.shuffle(&mut team1.programs);
    rng.shuffle(&mut team2.programs);

    let used_team2_ids: HashSet<u64> = HashSet::new();

    // avoid doing crossover that would result in empty programs
    team1.programs = team1
        .programs
        .iter()
        .map(|program| {
            let team1_action = program.action;

            for team2_program in team2.programs.iter_mut() {
                let team2_action = team2_program.action;
                if team1_action == team2_action && !used_team2_ids.contains(&team2.id) {
                    let original_program = program.clone();
                    let mut retry_count = 0;
                    loop {
                        let mut crossed_over_program = original_program.clone();
                        size_fair_dependent_instruction_crossover(
                            &mut crossed_over_program,
                            team2_program,
                            rng,
                            params,
                        );
                        crossed_over_program.mark_introns(params);

                        if !crossed_over_program.active_instructions.is_empty() {
                            return crossed_over_program;
                        }
                        retry_count += 1;
                        if retry_count > MAX_MUTATION_CROSSOVER_ATTEMPTS {
                            return program.clone();
                        }
                    }
                }
            }
            // if we can't find a viable crossover point, just return the original program
            program.clone()
        })
        .collect();
}

type IndividualErrorFunction<A, Fitness> = fn(
    &Team<A, Fitness>,
    &[Vec<f32>],
    &ProblemParameters<Fitness>,
    &[A],
) -> (Fitness, BehaviorDescriptor);

fn evaluate_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    teams: &mut Vec<Team<A, Fitness>>,
    fitness_cases: &[Vec<f32>],
    labels: &[A],
    individual_output: IndividualErrorFunction<A, Fitness>,
    params: &ProblemParameters<Fitness>,
) {
    if EVALUATE_PARALLEL {
        teams.par_iter_mut().for_each(|team| {
            // tracking skipped evaluations the way we do in c++ code is not very helpful now
            if team.fitness.is_none() {
                let output = individual_output(team, fitness_cases, params, labels);
                team.fitness = Some(output.0);
                team.behavior_descriptor = Some(output.1);
            }
        });
        // todo archive entries and compute novelty stores here
    } else {
        for team in teams.iter_mut() {
            if team.fitness.is_none() {
                let output = individual_output(team, fitness_cases, params, labels);
                team.fitness = Some(output.0);
                team.behavior_descriptor = Some(output.1);
            }
        }
    }
}

struct RunParameters<
    'a,
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    run: usize,
    rng: &'a mut Rng,
    fitness_cases: &'a [Vec<f32>],
    labels: &'a [A],
    problem_parameters: &'a ProblemParameters<Fitness>,
    individual_output: IndividualErrorFunction<A, Fitness>,
    index_to_program_action: fn(usize) -> A,
    id_counter: &'a mut u64,
    dump: bool,
    seed: u64,
}

fn one_run<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    run_parameters: &mut RunParameters<A, Fitness>,
) -> Team<A, Fitness> {
    println!("Starting run # {}", run_parameters.run);

    println!("Initializing teams...");

    let mut teams = initialize_teams(
        run_parameters.rng,
        run_parameters.id_counter,
        run_parameters.problem_parameters,
        run_parameters.index_to_program_action,
    );
    println!("Done.");

    let mut best_team = teams[0].clone();

    let mut optimal_team_found = false;

    let mut stagnation_count = 0;

    for generation in 1..=run_parameters.problem_parameters.generation_count {
        println!("Starting generation {}", generation);
        evaluate_teams(
            &mut teams,
            run_parameters.fitness_cases,
            run_parameters.labels,
            run_parameters.individual_output,
            run_parameters.problem_parameters,
        );

        teams.sort_by(|team1, team2| {
            match team1
                .fitness
                .unwrap()
                .partial_cmp(&team2.fitness.unwrap())
                .unwrap()
            {
                Ordering::Equal => team1
                    .active_instruction_count()
                    .partial_cmp(&team2.active_instruction_count())
                    .unwrap(),
                other => other,
            }
        });

        if run_parameters.dump {
            fs::create_dir_all(format!("dump/{}", run_parameters.seed)).unwrap();
            let output_path = format!("dump/{}/{}.txt", run_parameters.seed, generation);
            let mut file = File::create(output_path).unwrap();

            for team in teams.iter() {
                write!(file, "Fitness {}:\n{}", team.fitness.unwrap(), team).unwrap();
            }
        }

        let mut new_best_found = false;

        for team in teams.iter() {
            if best_team.fitness.is_none() || (team.fitness.unwrap() < best_team.fitness.unwrap()) {
                new_best_found = true;
                best_team = team.clone();
                println!(
                    "New best found in run #{}, generation {}, with fitness {} and active instruction count {}:",
                    run_parameters.run,
                    generation,
                    team.fitness.unwrap(),
                    team.active_instruction_count()
                );
                println!("{}", team);
            }

            if team.fitness.unwrap() <= run_parameters.problem_parameters.fitness_threshold {
                optimal_team_found = true;
                println!(
                    "Optimal found in run #{}, generation {}, with fitness {}:",
                    run_parameters.run,
                    generation,
                    team.fitness.unwrap()
                );
                println!("{}", team);
                break;
            }
        }

        if optimal_team_found {
            break;
        }

        if new_best_found {
            stagnation_count = 0;
        } else {
            stagnation_count += 1;
            println!("Stagnation count has increased to {}", stagnation_count);
        }

        if stagnation_count
            > run_parameters
                .problem_parameters
                .generation_stagnation_limit
        {
            println!(
                "Stagnation count exceeds limit of {}, exiting",
                run_parameters
                    .problem_parameters
                    .generation_stagnation_limit
            );
            break;
        }

        // TODO calculate and display min/max/median fitness

        let mut next_generation = teams.clone();
        next_generation.truncate(run_parameters.problem_parameters.deletion_point());

        while next_generation.len() < run_parameters.problem_parameters.population_size {
            loop {
                let parent1_index = tournament_selection_fitness(
                    &teams,
                    run_parameters.rng,
                    run_parameters.problem_parameters,
                );
                let mut parent1 = teams[parent1_index].clone();

                let previous_team = parent1.clone();

                parent1.restore_introns();

                let parent2_index = tournament_selection_fitness(
                    &teams,
                    run_parameters.rng,
                    run_parameters.problem_parameters,
                );
                let mut parent2 = teams[parent2_index].clone();

                parent2.restore_introns();

                let parent1_id = parent1.id;
                let parent2_id = parent2.id;

                if parent1.programs.len() <= parent2.programs.len() {
                    team_crossover(
                        &mut parent1,
                        &mut parent2,
                        run_parameters.rng,
                        run_parameters.problem_parameters,
                    );
                } else {
                    team_crossover(
                        &mut parent2,
                        &mut parent1,
                        run_parameters.rng,
                        run_parameters.problem_parameters,
                    );
                    parent1 = parent2.clone();
                }

                mutate_team(
                    &mut parent1,
                    &mut parent2,
                    run_parameters.rng,
                    run_parameters.id_counter,
                    run_parameters.problem_parameters,
                    run_parameters.index_to_program_action,
                );
                parent1.mark_introns(run_parameters.problem_parameters);

                if previous_team != parent1 {
                    *run_parameters.id_counter += 1;
                    parent1.id = *run_parameters.id_counter;
                    parent1.parent1_id = parent1_id;
                    parent1.parent2_id = parent2_id;
                    parent1.fitness = None;
                    // only add new individuals
                    next_generation.push(parent1);
                    break;
                }
            }
        }
        teams = next_generation.clone();
    }

    println!("Done with run # {}", run_parameters.run);

    best_team
}

pub fn get_seed_value() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}

fn fitness_case_with_constants<
    Fitness: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    inputs: Vec<f32>,
    params: &ProblemParameters<Fitness>,
) -> Vec<f32> {
    let mut output = vec![0.0; params.fitness_case_size()];

    for (i, input) in inputs.iter().enumerate() {
        output[i] = *input;
    }

    for (i, constant) in params.constant_list.iter().enumerate() {
        let target_index = i + params.input_count;
        output[target_index] = *constant;
    }

    output
}
