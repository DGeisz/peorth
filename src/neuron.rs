use std::cell::RefCell;
use std::rc::Rc;

#[derive(Copy, Clone)]
pub enum ChargeCycle {
    Even,
    Odd,
}

impl ChargeCycle {
    // Note that prev_cycle and next_cycle
    // do the exact same thing
    pub fn next_cycle(&self) -> ChargeCycle {
        self.prev_cycle()
    }

    pub fn prev_cycle(&self) -> ChargeCycle {
        match self {
            ChargeCycle::Even => ChargeCycle::Odd,
            ChargeCycle::Odd => ChargeCycle::Even,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Measure {
    value: f32,
    quality: f32, // Value between 0 and 1 that essentially is greater if the value is more deterministic
}

impl Measure {
    pub fn new(value: f32, quality: f32) -> Measure {
        Measure { value, quality }
    }

    pub fn get_weighted_value(&self) -> f32 {
        self.value * self.quality
    }
}

pub trait Neuronic {
    fn run_cycle(&self, cycle: ChargeCycle);
}

pub trait NeuronicInput {
    fn get_measure(&self, cycle: ChargeCycle) -> Measure;
    fn intake_external_measure(&self, cycle: ChargeCycle, measure: Measure);
}

/// Stores the Neuron's measure for different charge cycles
pub struct NeuronCharge(f32, f32);

impl NeuronCharge {
    pub fn new() -> NeuronCharge {
        NeuronCharge(0.0, 0.0)
    }

    pub fn set_measure(&mut self, cycle: ChargeCycle, measure: f32) {
        match cycle {
            ChargeCycle::Even => self.0 = measure,
            ChargeCycle::Odd => self.1 = measure,
        }
    }

    pub fn get_measure(&self, cycle: ChargeCycle) -> f32 {
        match cycle {
            ChargeCycle::Even => self.0,
            ChargeCycle::Odd => self.1,
        }
    }
}

pub struct ExternalWeightedCharge {
    weighted_value: f32,
    total_weight: f32,
    num_charges: f32,
}

impl ExternalWeightedCharge {
    pub fn new() -> ExternalWeightedCharge {
        ExternalWeightedCharge {
            weighted_value: 0.0,
            total_weight: 0.0,
            num_charges: 0.0,
        }
    }

    pub fn add_measure(&mut self, measure: Measure) {
        self.weighted_value += measure.value * measure.quality;
        self.total_weight += measure.quality;
        self.num_charges += 1.;
    }

    pub fn get_weighted_measure(&self) -> Measure {
        Measure::new(
            self.weighted_value / self.total_weight,
            self.total_weight / self.num_charges,
        )
    }

    pub fn clear(&mut self) {
        self.weighted_value = 0.0;
        self.total_weight = 0.0;
        self.num_charges = 0.0;
    }
}

pub struct ExternalCharge(ExternalWeightedCharge, ExternalWeightedCharge);

impl ExternalCharge {
    pub fn new() -> ExternalCharge {
        ExternalCharge(ExternalWeightedCharge::new(), ExternalWeightedCharge::new())
    }

    pub fn add_measure(&mut self, cycle: ChargeCycle, measure: Measure) {
        match cycle {
            ChargeCycle::Even => {
                self.0.add_measure(measure);
            }
            ChargeCycle::Odd => {
                self.1.add_measure(measure);
            }
        }
    }

    pub fn get_weighted_measure(&self, cycle: ChargeCycle) -> Measure {
        match cycle {
            ChargeCycle::Even => self.0.get_weighted_measure(),
            ChargeCycle::Odd => self.1.get_weighted_measure(),
        }
    }

    pub fn clear_cycle(&mut self, cycle: ChargeCycle) {
        match cycle {
            ChargeCycle::Even => {
                self.0.clear();
            }
            ChargeCycle::Odd => {
                self.1.clear();
            }
        }
    }

    pub fn clear_all(&mut self) {
        self.0.clear();
        self.1.clear();
    }
}

pub struct InteralWeightedCharge {
    weighted_measure: f32,
    weighted_prediction_quality: f32,
    total_measure_quality: f32,
}

impl InteralWeightedCharge {
    pub fn new() -> InteralWeightedCharge {
        InteralWeightedCharge {
            weighted_measure: 0.0,
            weighted_prediction_quality: 0.0,
            total_measure_quality: 0.0,
        }
    }

    pub fn add_measure(&mut self, measure: Measure, prediction_quality: f32) {
        self.weighted_measure += measure.quality * prediction_quality * measure.quality;
        self.weighted_prediction_quality += prediction_quality * measure.quality;
        self.total_measure_quality += measure.quality;
    }

    pub fn get_measure(&self) -> Measure {
        Measure::new(
            self.weighted_measure / self.weighted_prediction_quality,
            self.weighted_prediction_quality / self.total_measure_quality,
        )
    }

    pub fn clear(&mut self) {
        self.weighted_measure = 0.0;
        self.weighted_prediction_quality = 0.0;
        self.total_measure_quality = 0.0;
    }
}

pub struct InternalCharge(InteralWeightedCharge, InteralWeightedCharge);

impl InternalCharge {
    pub fn new() -> InternalCharge {
        InternalCharge(InteralWeightedCharge::new(), InteralWeightedCharge::new())
    }

    pub fn add_measure(&mut self, cycle: ChargeCycle, measure: Measure, prediction_quality: f32) {
        match cycle {
            ChargeCycle::Even => {
                self.0.add_measure(measure, prediction_quality);
            },
            ChargeCycle::Odd => {
                self.1.add_measure(measure, prediction_quality);
            }
        }
    }

    pub fn get_measure(&self, cycle: ChargeCycle) -> Measure {
        match cycle {
            ChargeCycle::Even => {
                self.0.get_measure()
            },
            ChargeCycle::Odd => {
                self.1.get_measure()
            }
        }
    }

    pub fn clear_cycle(&mut self, cycle: ChargeCycle) {
        match cycle {
            ChargeCycle::Even => {
                self.0.clear();
            },
            ChargeCycle::Odd => {
                self.1.clear();
            }
        }
    }

    pub fn clear_all(&mut self) {
        self.0.clear();
        self.1.clear();
    }
}

pub struct Synapse {
    input: Rc<dyn NeuronicInput>,
    ema: f32
}


pub struct Neuron {
    synapses: RefCell<Vec<Rc<dyn NeuronicInput>>>,
    prediction_weights: RefCell<Vec<Vec<f32>>>,
    external_charge: RefCell<ExternalCharge>,
    internal_charge: RefCell<InternalCharge>,
    neuron_charge: RefCell<NeuronCharge>, //Takes internal and external charge into account
    learning_rate: f32,
    ema_alpha: f32,
    get_quality: fn(value: f32, target: f32) -> f32 // Should return between 0 and 1
}

impl Neuron {
    /// This simply instantiates a new neuron.  You must also call
    /// add_synapses to make this neuron do anything meaningful
    pub fn new(
        learning_rate: f32,
        ema_alpha: f32,
        get_quality: fn(value: f32, target: f32) -> f32
    ) -> Neuron {
        Neuron {
            synapses: RefCell::new(Vec::new()),
            prediction_weights: RefCell::new(Vec::new()),
            external_charge: RefCell::new(ExternalCharge::new()),
            internal_charge: RefCell::new(InternalCharge::new()),
            neuron_charge: RefCell::new(NeuronCharge::new()),
            learning_rate,
            ema_alpha,
            get_quality
        }
    }

    pub fn add_synapses(
        &self,
        synapses: Vec<Rc<dyn NeuronicInput>>,
        initial_prediction_weights: Vec<Vec<f32>>,
    ) {
        //Make sure everything lines up properly
        if synapses.len() != initial_prediction_weights.len() {
            panic!("Number of synapses does not match length of prediction weights vector");
        }

        for (i, weights_vector) in initial_prediction_weights.iter().enumerate() {
            if weights_vector.len() != synapses.len() - 1 {
                panic!(
                    "Weights vector {} has length {}, should have length {}",
                    i,
                    weights_vector.len(),
                    synapses.len() - 1
                );
            }
        }

        *self.synapses.borrow_mut() = synapses;
        *self.prediction_weights.borrow_mut() = initial_prediction_weights;
    }

    pub fn intake_internal_charge(&self, cycle: ChargeCycle, measure: Measure, prediction_quality: f32) {
        self.internal_charge.borrow_mut().add_measure(cycle, measure, prediction_quality);
    }

    pub fn run_synapse_cycle(&self, cycle: ChargeCycle, synapse_index: usize, synapse_measure: Measure, neuron_weighted_measures: &Vec<f32>) {
        let mut prediction_weights = self.prediction_weights.borrow_mut();
        let mut weight_vector = prediction_weights.get_mut(synapse_index).unwrap();

        let mut prediction = 0.0;

        // Get prediction
        for (i, weighted_measure) in neuron_weighted_measures.iter().enumerate() {
            if i != synapse_index {
                let weight_index = if i < synapse_index { i } else { i - 1 };
                prediction += *weighted_measure * weight_vector[weight_index];
            }
        }

        let prediction_quality = (self.get_quality)(synapse_measure.value, prediction);

        // Intake the charge
        self.intake_internal_charge(cycle, synapse_measure, prediction_quality);

        // Modify weights along the gradient

        // Make sure we're moving the prediction in the right direction along gradient
        let learning_modifier = if prediction < synapse_measure.value { 1. } else { -1. };

        // Weight the learning rate so learning is smaller for less deterministic values (not worth it to try learning unstable values)
        let learning_constant = learning_modifier * self.learning_rate * synapse_measure.quality;

        // Update weights
        for (i, weighted_measure) in neuron_weighted_measures.iter().enumerate() {
            if i != synapse_index {
                let weight_index = if i < synapse_index { i } else { i - 1 };
                weight_vector[weight_index] += *weighted_measure * learning_constant;
            }
        }
    }
}

impl Neuronic for Neuron {
    fn run_cycle(&self, cycle: ChargeCycle) {
        let synapse_measures = self.synapses.borrow().iter().map(|synapse| synapse.get_measure(cycle)).collect();
    }
}
