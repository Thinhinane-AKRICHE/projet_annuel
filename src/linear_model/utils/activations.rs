pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(output: f64) -> f64 {
    1.0 - output.powi(2)
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn relu_derivative(output: f64) -> f64 {
    if output > 0.0 { 1.0 } else { 0.0 }
}

pub fn softmax(scores: &Vec<f64>) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();

    // éviter une division par zéro (rare mais possible)
    if sum_exp == 0.0 || !sum_exp.is_finite() {
        return vec![1.0 / scores.len() as f64; scores.len()]; // distribution uniforme
    }

    exp_scores.iter().map(|v| v / sum_exp).collect()
}



