mod acrobot;
mod wine_quality_classification;

// use crate::acrobot::AcrobotAction;
use clap::{App, Arg};
use fastrand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::iter::FromIterator;
use std::time::{SystemTime, UNIX_EPOCH};

fn coin_flip(p: f32, rng: &mut Rng) -> bool {
    rng.f32() < p
}

fn random_float_in_range(rng: &mut Rng, lower: f32, upper: f32) -> f32 {
    lower + (upper - lower) * rng.f32()
}

const EVALUATE_PARALLEL: bool = true;

// this can be treated as problem-independent even though a given problem might pick a subset where
// the actual effective max arity is lower. in general, this code views the function set as relatively
// problem independent so the slight waste in space in each instruction is something that can be dealt with later
const MAX_ARITY: usize = 3;

pub struct ProblemParameters {
    input_count: usize,
    register_count: usize,
    approximate_fitness_case_count: usize,
    population_size: usize,
    population_to_delete: usize,
    max_program_size: usize,
    min_initial_program_size: usize,
    max_initial_program_size: usize,
    action_count: usize,
    max_initial_team_size: usize,
    max_team_size: usize,
    generation_count: usize,
    generation_stagnation_limit: usize,
    run_count: usize,
    p_delete_instruction: f32,
    p_add_instruction: f32,
    p_swap_instructions: f32,
    p_change_destination: f32,
    p_change_function: f32,
    p_change_input: f32,
    p_flip_input: f32,
    p_change_action: f32,
    p_delete_program: f32,
    p_add_program: f32,
    fitness_threshold: f32,
    legal_functions: Vec<Function>,
    constant_list: Vec<f32>,
}

impl ProblemParameters {
    fn deletion_point(&self) -> usize {
        self.population_size - self.population_to_delete
    }

    fn fitness_case_size(&self) -> usize {
        self.input_count + self.constant_list.len()
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
enum Function {
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

#[derive(Debug, Clone, Copy)]
struct Instruction {
    destination: usize,
    operands: [usize; MAX_ARITY],
    is_register: [bool; MAX_ARITY],
    op: Function,
    index: usize,
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

fn random_instruction(rng: &mut Rng, params: &ProblemParameters) -> Instruction {
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

fn active_instructions_from_index(
    instructions: &[Instruction],
    starting_index: usize,
    params: &ProblemParameters,
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

#[derive(Debug, Clone)]
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

    fn mark_introns(&mut self, params: &ProblemParameters) {
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
>(
    program: &Program<A>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters,
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
>(
    parent1: &mut Program<A>,
    parent2: &mut Program<A>,
    rng: &mut Rng,
    params: &ProblemParameters,
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
>(
    program: &mut Program<A>,
    team_actions: &[A],
    rng: &mut Rng,
    counter: &mut u64,
    params: &ProblemParameters,
    index_to_program_action: fn(usize) -> A,
) {
    if program.active_instructions.is_empty() {
        return;
    }

    // only run one mutation per program at a time.
    if program.active_instructions.len() > 1 && coin_flip(params.p_delete_instruction, rng) {
        let delete_index = rng.usize(..program.active_instructions.len());
        program.active_instructions.remove(delete_index);
    } else if program.active_instructions.len() < params.max_program_size
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

#[derive(Debug, Clone)]
pub struct Team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
> {
    programs: Vec<Program<A>>,
    fitness: Option<f32>,
    behavior: Vec<A>,
    id: u64,
    parent1_id: u64,
    parent2_id: u64,
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > Team<A>
{
    fn restore_introns(&mut self) {
        for program in self.programs.iter_mut() {
            program.restore_introns();
        }
    }

    fn mark_introns(&mut self, params: &ProblemParameters) {
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
    > PartialEq for Team<A>
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
    > Eq for Team<A>
{
}

impl<
        A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
    > fmt::Display for Team<A>
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
>(
    rng: &mut Rng,
    id_counter: &mut u64,
    params: &ProblemParameters,
    index_to_program_action: fn(usize) -> A,
) -> Vec<Team<A>> {
    let mut teams = vec![];
    teams.reserve(params.population_size);

    while teams.len() < params.population_size {
        let mut team = Team {
            programs: vec![],
            fitness: None,
            behavior: vec![],
            id: 0,
            parent1_id: 0,
            parent2_id: 0,
        };
        team.programs.reserve(params.max_team_size);
        team.behavior.reserve(params.approximate_fitness_case_count);

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

fn evaluate_team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    team: &Team<A>,
    fitness_cases: &[Vec<f32>],
    params: &ProblemParameters,
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

fn mutate_team<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    team: &mut Team<A>,
    other_team: &mut Team<A>,
    rng: &mut Rng,
    counter: &mut u64,
    params: &ProblemParameters,
    index_to_program_action: fn(usize) -> A,
) {
    let team_actions: Vec<_> = team.programs.iter().map(|p| p.action).collect();

    for program in team.programs.iter_mut() {
        mutate_program(
            program,
            &team_actions,
            rng,
            counter,
            params,
            index_to_program_action,
        );
    }

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
>(
    team1: &mut Team<A>,
    team2: &mut Team<A>,
    rng: &mut Rng,
    params: &ProblemParameters,
) {
    assert!(team1.programs.len() <= team2.programs.len());

    // shuffle programs so that we can iterate in order without bias
    rng.shuffle(&mut team1.programs);
    rng.shuffle(&mut team2.programs);

    let used_team2_ids: HashSet<u64> = HashSet::new();

    for team1_program in team1.programs.iter_mut() {
        let team1_action = team1_program.action;

        for team2_program in team2.programs.iter_mut() {
            let team2_action = team2_program.action;
            if team1_action == team2_action && !used_team2_ids.contains(&team2.id) {
                size_fair_dependent_instruction_crossover(
                    team1_program,
                    team2_program,
                    rng,
                    params,
                );
            }
        }
    }
}

fn evaluate_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    teams: &mut Vec<Team<A>>,
    fitness_cases: &[Vec<f32>],
    labels: &[A],
    individual_error: fn(&Team<A>, &[Vec<f32>], &ProblemParameters, &[A]) -> (f32, Vec<A>),
    params: &ProblemParameters,
) {
    if EVALUATE_PARALLEL {
        teams.par_iter_mut().for_each(|team| {
            if team.fitness.is_none() {
                let (fitness, behavior) = individual_error(team, fitness_cases, params, labels);
                team.fitness = Some(fitness);
                team.behavior = behavior;
            }
        });
    } else {
        for team in teams.iter_mut() {
            if team.fitness.is_none() {
                let (fitness, behavior) = individual_error(team, fitness_cases, params, labels);
                team.fitness = Some(fitness);
                team.behavior = behavior;            }
        }
    }
}

fn one_run<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(
    run: usize,
    rng: &mut Rng,
    fitness_cases: &[Vec<f32>],
    labels: &[A],
    params: &ProblemParameters,
    individual_error: fn(&Team<A>, &[Vec<f32>], &ProblemParameters, &[A]) -> (f32, Vec<A>),
    index_to_program_action: fn(usize) -> A,
    id_counter: &mut u64,
    dump: bool,
    seed: u64,
) -> Team<A> {
    println!("Starting run # {}", run);

    let mut teams = initialize_teams(rng, id_counter, params, index_to_program_action);
    let mut best_team = teams[0].clone();
    let mut optimal_team_found = false;

    let mut archive = HashMap::new();
    let mut stagnation_count = 0;

    // TODO investigate what's going on here. check the archive, see what kind of behavior is occurring.
    // print stuff out and look it over.
    // dump data and investigate the population.
    // try acrobot and implement lawnmower/ant trail problems.
    // don't lose sight of goal: forex with memory.
    // try tweaking hyperparameters as well.
    for generation in 1..=params.generation_count {
        println!("Starting generation {}", generation);
        evaluate_teams(&mut teams, fitness_cases, labels, individual_error, params);

        teams = teams.into_iter().filter(|team| !archive.contains_key(&team.behavior)).collect();

        println!("Inserting {} teams into the archive.", teams.len());

        // all teams are novel, so insert them into the archive
        for team in teams.iter() {
            if !archive.contains_key(&team.behavior) {
                archive.insert(team.behavior.clone(), team.clone());
            }
        }

        println!("Archive size is now {}.", archive.len());

        for key in archive.keys() {
            println!("key = {:?}", key);
        }
        if dump {
            fs::create_dir_all(format!("dump/{}", seed)).unwrap();
            let output_path = format!("dump/{}/{}.txt", seed, generation);
            let mut file = File::create(output_path).unwrap();

            for team in teams.iter() {
                write!(file, "Fitness {}:\n{}", team.fitness.unwrap(), team).unwrap();
            }
        }

        for team in teams.iter() {
            if best_team.fitness.is_none() || (team.fitness.unwrap() < best_team.fitness.unwrap()) {
                best_team = team.clone();
                println!(
                    "New best found in run #{}, generation {}, with fitness {} and active instruction count {}:",
                    run,
                    generation,
                    team.fitness.unwrap(),
                    team.active_instruction_count()
                );
                println!("{}", team);
            }

            if team.fitness.unwrap() < params.fitness_threshold {
                optimal_team_found = true;
                println!(
                    "Optimal found in run #{}, generation {}, with fitness {}:",
                    run,
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

        if teams.is_empty() {
            stagnation_count += 1;
            println!("Teams list is empty. Stagnation count now at {}.", stagnation_count);

            if stagnation_count > params.generation_stagnation_limit {
                println!(
                    "Stagnation count exceeds limit of {}, exiting",
                    params.generation_stagnation_limit
                );
                break;
            }
        } else {
            stagnation_count = 0;
        }

        // keep breeding pool at a certain expected size
        if teams.len() < params.deletion_point() {
            let amount_to_copy = params.deletion_point() - teams.len();
            println!("Have to copy {} values from the archive", amount_to_copy);
            // copy from the archive and try again
            let archive_size = archive.len();
            let mut potential_archive_key_indexes: Vec<usize> = (0..archive_size).collect();
            rng.shuffle(&mut potential_archive_key_indexes);
            let taken_indexes: Vec<usize> = potential_archive_key_indexes.into_iter().take(amount_to_copy).collect();
            let chosen_archive_keys: HashSet<usize> = HashSet::from_iter(taken_indexes.into_iter());
            // since we chose keys randomly, it doesn't matter if the hashmap is ordered internally or not
            let archive_keys_to_copy: Vec<Vec<A>> = archive.keys().enumerate().filter(|(i, _k)|
                chosen_archive_keys.contains(i)
            ).map(|(_i, k)| k.to_vec()).collect();

            for key in archive_keys_to_copy.iter() {
                teams.push(archive.get(key).unwrap().clone());
            }
        }

        // Mutate existing teams in order to produce new behavior. Shuffle the population and mate
        // in pairs.
        rng.shuffle(&mut teams);

        // Mutate existing population here. Slightly awkward since mutating each team requires a second
        // team due to current definition of mutation
        for i in (0..teams.len()-2) {
            let mut parent1 = teams[i].clone();
            let original_parent1 = parent1.clone();

            let mut parent2 = teams[i+1].clone();
            parent1.restore_introns();
            parent2.restore_introns();

            if parent1.programs.len() <= parent2.programs.len() {
                team_crossover(&mut parent1, &mut parent2, rng, params);
            } else {
                team_crossover(&mut parent2, &mut parent1, rng, params);
                parent1 = parent2.clone();
            }

            // mutate until team has an effective change
            loop {
                mutate_team(
                    &mut parent1,
                    &mut parent2,
                    rng,
                    id_counter,
                    params,
                    index_to_program_action,
                );
                parent1.mark_introns(params);

                if parent1 != original_parent1 {
                    break;
                }
            }

            parent1.fitness = None;
            teams[i] = parent1.clone();
        }

        while teams.len() < params.population_size {
            loop {
                // all novel programs get to breed freely without regard to fitness
                let parent1_index = rng.usize(..teams.len());
                let mut parent1 = teams[parent1_index].clone();

                let previous_team = parent1.clone();

                parent1.restore_introns();

                let parent2_index = rng.usize(..teams.len());
                let mut parent2 = teams[parent2_index].clone();

                parent2.restore_introns();

                let parent1_id = parent1.id;
                let parent2_id = parent2.id;

                if parent1.programs.len() <= parent2.programs.len() {
                    team_crossover(&mut parent1, &mut parent2, rng, params);
                } else {
                    team_crossover(&mut parent2, &mut parent1, rng, params);
                    parent1 = parent2.clone();
                }

                mutate_team(
                    &mut parent1,
                    &mut parent2,
                    rng,
                    id_counter,
                    params,
                    index_to_program_action,
                );
                parent1.mark_introns(params);

                if previous_team != parent1 {
                    *id_counter += 1;
                    parent1.id = *id_counter;
                    parent1.parent1_id = parent1_id;
                    parent1.parent2_id = parent2_id;
                    parent1.fitness = None;
                    // only add new individuals
                    teams.push(parent1);
                    break;
                }
            }
        }
    }

    println!("Done with run # {}", run);

    best_team
}

fn get_seed_value() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as u64
}

fn fitness_case_with_constants(inputs: Vec<f32>, params: &ProblemParameters) -> Vec<f32> {
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

fn print_best_teams<
    A: Debug + Ord + PartialOrd + Eq + PartialEq + Hash + Copy + Clone + Display + Send + Sync,
>(best_teams: Vec<Team<A>>) {
    println!("Best teams:");

    for best_team in best_teams.iter() {
        println!("Fitness {}:", best_team.fitness.unwrap());
        println!("{}\n", best_team);
    }
}

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

fn main() {
    let (seed, dump, mut rng) = setup();

    let best_teams = acrobot::acrobot_runs(seed, dump, &mut rng);
    print_best_teams(best_teams);

    // let best_teams = wine_quality_classification::wine_runs(seed, dump, &mut rng);
    // print_best_teams(best_teams);

    println!("Ran with seed {}", seed);
}
