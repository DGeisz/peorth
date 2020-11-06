use crate::neuron::{get_quality, Neuron, NeuronicInput, NeuronicSensor, Neuronic, ChargeCycle};
use std::rc::Rc;

/// Utility method that compares f32 to
/// three decimal places
fn cmp_f32(f1: f32, f2: f32) {
    assert_eq!(
        (f1 * 1000.).floor(),
        (f2 * 1000.).floor(),
        "{} does not equal {}",
        f1,
        f2
    );
}

#[test]
fn test_basic_neuron() {
    let s1_value = 0.5;
    let s1 = Rc::new(NeuronicSensor::new_custom_start(s1_value));

    let s2_value = 0.6;
    let s2 = Rc::new(NeuronicSensor::new_custom_start(s2_value));

    let s2_w1 = 1.;
    let s1_w2 = 1.;
    let weights_vec = vec![vec![s1_w2], vec![s2_w1]];

    let input_vec = vec![
        Rc::clone(&s1) as Rc<dyn NeuronicInput>,
        Rc::clone(&s2) as Rc<dyn NeuronicInput>,
    ];

    let learning_rate = 0.01;

    let n = Neuron::new(learning_rate, get_quality);

    n.add_synapses(input_vec, weights_vec);

    let cycle = ChargeCycle::Even;
    n.run_cycle(cycle);

    let prediction_weights = n.prediction_weights.borrow();

    cmp_f32(
        *prediction_weights.get(0).unwrap().get(0).unwrap(),
        s2_w1 - s2_value * learning_rate
    );

    cmp_f32(
        *prediction_weights.get(1).unwrap().get(0).unwrap(),
        s1_w2 + s1_value * learning_rate
    )
}
