// seed 1623545378569 finds a 3-instruction solution.

use clap::{App, Arg};
use fastrand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::fmt::Formatter;
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

const EVALUATE_PARALLEL: bool = true;

const INPUT_COUNT: usize = 4;
const FITNESS_CASE_COUNT: usize = 100;
const REGISTER_COUNT: usize = 4;
const POPULATION_SIZE: usize = 10000;
const POPULATION_TO_DELETE: usize = 9500;
const DELETION_POINT: usize = POPULATION_SIZE - POPULATION_TO_DELETE;
const MAX_PROGRAM_SIZE: usize = 32;
const MIN_INITIAL_PROGRAM_SIZE: usize = 1;
const MAX_INITIAL_PROGRAM_SIZE: usize = 12;
const ACTION_COUNT: usize = 3;
const MAX_INITIAL_TEAM_SIZE: usize = ACTION_COUNT * 3;
const MAX_TEAM_SIZE: usize = ACTION_COUNT * 6;

const TOURNAMENT_SIZE: usize = 4;
const GENERATION_COUNT: usize = 1000;
const GENERATION_STAGNATION_LIMIT: usize = 25;
const RUN_COUNT: usize = 1;

const P_DELETE_INSTRUCTION: f32 = 0.8;
const P_ADD_INSTRUCTION: f32 = 0.8;
const P_SWAP_INSTRUCTIONS: f32 = 0.8;
const P_CHANGE_DESTINATION: f32 = 0.1;
const P_CHANGE_FUNCTION: f32 = 0.1;
const P_CHANGE_INPUT: f32 = 0.1;
const P_FLIP_INPUT: f32 = 0.1;
const P_CHANGE_ACTION: f32 = 0.1;

const P_DELETE_PROGRAM: f32 = 0.5;
const P_ADD_PROGRAM: f32 = 0.5;

const FITNESS_THRESHOLD: f32 = 45.0 * (FITNESS_CASE_COUNT as f32) + 1.0;

const MAX_ARITY: usize = 3;

const NEGATIVE_TORQUE: f32 = -1.0;
const POSITIVE_TORQUE: f32 = 1.0;

const FITNESS_CASE_SIZE: usize = INPUT_COUNT + CONSTANT_COUNT;
type FitnessCase = [f32; FITNESS_CASE_SIZE];

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone)]
enum ProgramAction {
    NegativeTorque,
    DoNothing,
    PositiveTorque,
}

fn index_to_program_action(index: usize) -> ProgramAction {
    match index {
        0 => ProgramAction::NegativeTorque,
        1 => ProgramAction::DoNothing,
        2 => ProgramAction::PositiveTorque,
        _ => panic!(),
    }
}

impl fmt::Display for ProgramAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramAction::NegativeTorque => write!(f, "Negative_Torque"),
            ProgramAction::DoNothing => write!(f, "Do_Nothing"),
            ProgramAction::PositiveTorque => write!(f, "Positive_Torque"),
        }
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

const LEGAL_FUNCTION_COUNT: usize = 18;

const LEGAL_FUNCTIONS: [Function; LEGAL_FUNCTION_COUNT] = [
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
];

const CONSTANT_COUNT: usize = 1;

// adding more constants seems to hurt things rather than help for acrobot
const CONSTANT_LIST: [f32; CONSTANT_COUNT] = [0.0];

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

fn random_instruction(rng: &mut Rng) -> Instruction {
    let mut instruction = Instruction {
        destination: 0,
        operands: [0; MAX_ARITY],
        is_register: [false; MAX_ARITY],
        op: Function::Relu,
        index: 0,
    };

    instruction.destination = rng.usize(..REGISTER_COUNT);

    let function_index = rng.usize(..LEGAL_FUNCTION_COUNT);
    let random_function = LEGAL_FUNCTIONS[function_index];
    instruction.op = random_function;
    let arity = function_arity(&random_function);

    for i in 0..arity {
        let is_register = rng.bool();
        instruction.is_register[i] = is_register;

        let index_range = if is_register {
            REGISTER_COUNT
        } else {
            FITNESS_CASE_SIZE
        };

        let input_index = rng.usize(..index_range);
        instruction.operands[i] = input_index;
    }

    instruction
}

fn active_instructions_from_index(
    instructions: &[Instruction],
    starting_index: usize,
) -> [bool; MAX_PROGRAM_SIZE] {
    let mut active_instructions = [false; MAX_PROGRAM_SIZE];

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
struct Program {
    active_instructions: Vec<Instruction>,
    introns: Vec<Instruction>,
    action: usize,
    id: u64,
}

impl Program {
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

    fn mark_introns(&mut self) {
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
            active_instructions_from_index(&self.active_instructions, last_output_index);

        let mut new_active_instructions = vec![];
        new_active_instructions.reserve(MAX_PROGRAM_SIZE);

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

impl Hash for Program {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        for instruction in self.active_instructions.iter() {
            instruction.hash(state);
        }
    }
}

impl PartialEq for Program {
    fn eq(&self, other: &Self) -> bool {
        self.action == other.action
            && self
                .active_instructions
                .iter()
                .zip(&other.active_instructions)
                .all(|(i1, i2)| i1 == i2)
    }
}

impl Eq for Program {}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "\t\tAction: {}", index_to_program_action(self.action)).unwrap();
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

fn evaluate_program(program: &Program, fitness_cases: &[FitnessCase]) -> Vec<f32> {
    let mut registers: Vec<[f32; REGISTER_COUNT]> =
        vec![[0.0; REGISTER_COUNT]; fitness_cases.len()];

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

fn size_fair_dependent_instruction_crossover(
    parent1: &mut Program,
    parent2: &mut Program,
    rng: &mut Rng,
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
    let parent1_active_indexes =
        active_instructions_from_index(&parent1.active_instructions, parent1_crossover_index);
    let parent1_subtree_indexes: Vec<usize> = parent1_active_indexes
        .iter()
        .enumerate()
        .filter(|(_, elt)| **elt)
        .map(|(index, _)| index)
        .collect();
    let parent1_active_instruction_count = parent1_subtree_indexes.len();

    let parent2_subtree_sizes = (0..parent2.active_instructions.len()).map(|index| {
        active_instructions_from_index(&parent2.active_instructions, index)
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
        active_instructions_from_index(&parent2.active_instructions, closest_subtree_index);
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

fn mutate_program(program: &mut Program, team_actions: &[usize], rng: &mut Rng, counter: &mut u64) {
    if program.active_instructions.is_empty() {
        return;
    }

    // only run one mutation per program at a time.
    if program.active_instructions.len() > 1 && coin_flip(P_DELETE_INSTRUCTION, rng) {
        let delete_index = rng.usize(..program.active_instructions.len());
        program.active_instructions.remove(delete_index);
    } else if program.active_instructions.len() < MAX_PROGRAM_SIZE
        && coin_flip(P_ADD_INSTRUCTION, rng)
    {
        let add_index = rng.usize(..=program.active_instructions.len());
        let instruction = random_instruction(rng);
        program.active_instructions.insert(add_index, instruction);
    } else if program.active_instructions.len() >= 2 && coin_flip(P_SWAP_INSTRUCTIONS, rng) {
        let index1 = rng.usize(..program.active_instructions.len());
        let index2 = rng.usize(..program.active_instructions.len());

        program.active_instructions.swap(index1, index2);
    } else if coin_flip(P_CHANGE_DESTINATION, rng) {
        let index = rng.usize(..program.active_instructions.len());
        let destination = rng.usize(..REGISTER_COUNT);
        program.active_instructions[index].destination = destination;
    } else if coin_flip(P_CHANGE_FUNCTION, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());
        let current_op = program.active_instructions[instruction_index].op;
        let current_arity = function_arity(&current_op);

        let equal_arity_functions: Vec<_> = LEGAL_FUNCTIONS
            .iter()
            .filter(|f| function_arity(f) == current_arity)
            .collect();

        let equal_arity_function_count = equal_arity_functions.len();

        let new_function_index = rng.usize(..equal_arity_function_count);
        let new_op = equal_arity_functions[new_function_index];
        program.active_instructions[instruction_index].op = *new_op;
    } else if coin_flip(P_FLIP_INPUT, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());
        let instruction_op = program.active_instructions[instruction_index].op;
        let arity = function_arity(&instruction_op);
        let input_index = rng.usize(..arity);

        let is_register = rng.bool();
        program.active_instructions[instruction_index].is_register[input_index] = is_register;

        let current_operand = program.active_instructions[instruction_index].operands[input_index];

        if current_operand >= REGISTER_COUNT && is_register {
            program.active_instructions[instruction_index].operands[input_index] =
                rng.usize(..REGISTER_COUNT);
        }
    } else if coin_flip(P_CHANGE_INPUT, rng) {
        let instruction_index = rng.usize(..program.active_instructions.len());

        let instruction_op = program.active_instructions[instruction_index].op;
        let arity = function_arity(&instruction_op);
        let input_index = rng.usize(..arity);

        let is_register = program.active_instructions[instruction_index].is_register[input_index];

        let limit = if is_register {
            REGISTER_COUNT
        } else {
            FITNESS_CASE_SIZE
        };

        program.active_instructions[instruction_index].operands[input_index] = rng.usize(..limit);
    } else if coin_flip(P_CHANGE_ACTION, rng) {
        let action_index = rng.usize(..ACTION_COUNT);

        let learners_with_action = team_actions
            .iter()
            .enumerate()
            .filter(|(program_index, action)| {
                *program_index != action_index && **action == action_index
            })
            .count();

        // only change action if there is another learner with this action so actions are not lost
        // if it is not beneficial to ever perform a certain action, learners will have to evolve
        // a no-op program
        if learners_with_action >= 1 {
            program.action = action_index;
        }
    }

    // effective instructions may have changed
    // program.mark_introns();
    *counter += 1;
    program.id = *counter;
}

#[derive(Debug, Clone)]
struct Team {
    programs: Vec<Program>,
    fitness: Option<f32>,
    id: u64,
    parent1_id: u64,
    parent2_id: u64,
}

impl Team {
    fn restore_introns(&mut self) {
        for program in self.programs.iter_mut() {
            program.restore_introns();
        }
    }

    fn mark_introns(&mut self) {
        for program in self.programs.iter_mut() {
            program.mark_introns();
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

impl PartialEq for Team {
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

impl Eq for Team {}

impl fmt::Display for Team {
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

fn initialize_teams(rng: &mut Rng, id_counter: &mut u64) -> Vec<Team> {
    let mut teams = vec![];
    teams.reserve(POPULATION_SIZE);

    while teams.len() < POPULATION_SIZE {
        let mut team = Team {
            programs: vec![],
            fitness: None,
            id: 0,
            parent1_id: 0,
            parent2_id: 0,
        };
        team.programs.reserve(MAX_TEAM_SIZE);

        let program_count = rng.usize(..MAX_INITIAL_TEAM_SIZE);

        let mut program_set = HashSet::new();
        program_set.reserve(MAX_TEAM_SIZE);

        // make sure each team has at least one of each action type
        for action_index in 0..ACTION_COUNT {
            loop {
                let mut program = Program {
                    active_instructions: vec![],
                    introns: vec![],
                    action: action_index,
                    id: 0,
                };
                program.active_instructions.reserve(MAX_PROGRAM_SIZE);
                program.introns.reserve(MAX_PROGRAM_SIZE);

                let instruction_count =
                    rng.usize(MIN_INITIAL_PROGRAM_SIZE..MAX_INITIAL_PROGRAM_SIZE);

                for _ in 0..instruction_count {
                    let instruction = random_instruction(rng);
                    program.active_instructions.push(instruction);
                }

                program.mark_introns();

                // clear introns so that initially all code is active
                program.introns.clear();
                // just to set indexes properly
                program.mark_introns();

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
            let action = rng.usize(..ACTION_COUNT);
            loop {
                let mut program = Program {
                    active_instructions: vec![],
                    introns: vec![],
                    action,
                    id: 0,
                };
                program.active_instructions.reserve(MAX_PROGRAM_SIZE);
                program.introns.reserve(MAX_PROGRAM_SIZE);

                let instruction_count =
                    rng.usize(MIN_INITIAL_PROGRAM_SIZE..MAX_INITIAL_PROGRAM_SIZE);

                for _ in 0..instruction_count {
                    let instruction = random_instruction(rng);
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

fn evaluate_team(team: &Team, fitness_cases: &[FitnessCase]) -> Vec<ProgramAction> {
    let team_outputs: Vec<Vec<f32>> = team
        .programs
        .iter()
        .map(|p| evaluate_program(p, fitness_cases))
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
        .map(index_to_program_action)
        .collect()
}

fn tournament_selection(teams: &[Team], rng: &mut Rng) -> usize {
    (0..TOURNAMENT_SIZE)
        .map(|_| rng.usize(..DELETION_POINT))
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

fn mutate_team(team: &mut Team, other_team: &mut Team, rng: &mut Rng, counter: &mut u64) {
    let team_actions: Vec<_> = team.programs.iter().map(|p| p.action).collect();

    for program in team.programs.iter_mut() {
        mutate_program(program, &team_actions, rng, counter);
    }

    if team.programs.len() < MAX_TEAM_SIZE && coin_flip(P_ADD_PROGRAM, rng) {
        let other_team_index = rng.usize(..other_team.programs.len());
        team.programs
            .push(other_team.programs[other_team_index].clone());
    }

    if team.programs.len() > 1 && coin_flip(P_DELETE_PROGRAM, rng) {
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

fn team_crossover(team1: &mut Team, team2: &mut Team, rng: &mut Rng) {
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
                size_fair_dependent_instruction_crossover(team1_program, team2_program, rng);
            }
        }
    }
}

fn evaluate_teams(teams: &mut Vec<Team>, fitness_cases: &[FitnessCase]) {
    if EVALUATE_PARALLEL {
        teams.par_iter_mut().for_each(|team| {
            // tracking skipped evaluations the way we do in c++ code is not very helpful now
            if team.fitness.is_none() {
                team.fitness = Some(individual_error(team, fitness_cases));
            }
        });
    } else {
        for team in teams.iter_mut() {
            if team.fitness.is_none() {
                team.fitness = Some(individual_error(team, fitness_cases));
            }
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

fn individual_error(team: &Team, fitness_cases: &[FitnessCase]) -> f32 {
    let mut steps: Vec<usize> = vec![0; FITNESS_CASE_COUNT];

    let mut total_steps = 0;

    let mut state = fitness_cases.to_owned();

    let episode_limit = 500;

    loop {
        if state.is_empty() {
            break;
        }

        let outputs = evaluate_team(team, &state);

        let mut to_delete = vec![false; outputs.len()];

        for (output_index, (current_state, output)) in state.iter_mut().zip(&outputs).enumerate() {
            let torque = match output {
                ProgramAction::NegativeTorque => NEGATIVE_TORQUE,
                ProgramAction::DoNothing => 0.0,
                ProgramAction::PositiveTorque => POSITIVE_TORQUE,
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

fn one_run(
    run: usize,
    rng: &mut Rng,
    fitness_cases: &[FitnessCase],
    id_counter: &mut u64,
    dump: bool,
    seed: u64,
) -> Team {
    println!("Starting run # {}", run);

    println!("Initializing teams...");

    let mut teams = initialize_teams(rng, id_counter);
    println!("Done.");

    let mut best_team = teams[0].clone();

    let mut optimal_team_found = false;

    let mut stagnation_count = 0;

    for generation in 1..=GENERATION_COUNT {
        println!("Starting generation {}", generation);
        evaluate_teams(&mut teams, fitness_cases);

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

        if dump {
            fs::create_dir_all(format!("dump/{}", seed)).unwrap();
            let output_path = format!("dump/{}/{}.txt", seed, generation);
            let mut file = File::create(output_path).unwrap();

            for team in teams.iter() {
                write!(file, "Fitness {}\n:{}", team.fitness.unwrap(), team).unwrap();
            }
        }

        // skipping file logging for now
        let mut new_best_found = false;

        for team in teams.iter() {
            if best_team.fitness.is_none() || (team.fitness.unwrap() < best_team.fitness.unwrap()) {
                new_best_found = true;
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

            if team.fitness.unwrap() < FITNESS_THRESHOLD {
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

        if new_best_found {
            stagnation_count = 0;
        } else {
            stagnation_count += 1;
            println!("Stagnation count has increased to {}", stagnation_count);
        }

        if stagnation_count > GENERATION_STAGNATION_LIMIT {
            println!(
                "Stagnation count exceeds limit of {}, exiting",
                GENERATION_STAGNATION_LIMIT
            );
            break;
        }

        // TODO calculate and display min/max/median fitness

        teams.truncate(DELETION_POINT);
        assert!(teams.len() == DELETION_POINT);

        while teams.len() < POPULATION_SIZE {
            loop {
                let parent1_index = tournament_selection(&teams, rng);
                let mut parent1 = teams[parent1_index].clone();

                let previous_team = parent1.clone();

                parent1.restore_introns();

                let parent2_index = tournament_selection(&teams, rng);
                let mut parent2 = teams[parent2_index].clone();

                parent2.restore_introns();

                let parent1_id = parent1.id;
                let parent2_id = parent2.id;

                if parent1.programs.len() <= parent2.programs.len() {
                    team_crossover(&mut parent1, &mut parent2, rng);
                } else {
                    team_crossover(&mut parent2, &mut parent1, rng);
                    parent1 = parent2.clone();
                }

                mutate_team(&mut parent1, &mut parent2, rng, id_counter);
                parent1.mark_introns();

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

fn fitness_case_with_constants(inputs: [f32; INPUT_COUNT]) -> FitnessCase {
    let mut output = [0.0; FITNESS_CASE_SIZE];

    for (i, input) in inputs.iter().enumerate() {
        output[i] = *input;
    }

    for (i, constant) in CONSTANT_LIST.iter().enumerate() {
        let target_index = i + INPUT_COUNT;
        output[target_index] = *constant;
    }

    output
}

fn main() {
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

    let mut rng = Rng::with_seed(seed);

    let mut id_counter: u64 = 1;

    let mut best_teams: Vec<Team> = vec![];

    let mut fitness_cases: Vec<FitnessCase> = vec![];

    for _ in 0..FITNESS_CASE_COUNT {
        let x1 = random_float_in_range(&mut rng, -0.1, 0.1);
        let v1 = random_float_in_range(&mut rng, -0.1, 0.1);
        let x2 = random_float_in_range(&mut rng, -0.1, 0.1);
        let v2 = random_float_in_range(&mut rng, -0.1, 0.1);

        fitness_cases.push(fitness_case_with_constants([x1, v1, x2, v2]));
    }

    for run in 1..=RUN_COUNT {
        best_teams.push(one_run(
            run,
            &mut rng,
            &fitness_cases,
            &mut id_counter,
            dump,
            seed,
        ));
    }

    println!("Best teams:");

    for best_team in best_teams.iter() {
        println!("Fitness {}:", best_team.fitness.unwrap());
        println!("{}\n", best_team);
    }

    println!("Ran with seed {}", seed);
}
