pub mod graph;

use graph::{load_dataset, build_graph, predict_risk, randomize_and_split, Patient};

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = load_dataset("Heart_OnlyNum .csv")?;
    println!("Loaded {} patients.", dataset.len());

    let test_ratio = 0.2;
    let (
        (train1, test1),
        _,
        _,
        _,
    ) = randomize_and_split(dataset, test_ratio);

    let (train_data, test_data) = (train1, test1);

    println!(
        "Selected quarter - Training data size: {}, Testing data size: {}",
        train_data.len(),
        test_data.len()
    );

    let threshold = 5;
    let graph = build_graph(&train_data, threshold);

    let mut correct_predictions = 0;
    for test_patient in &test_data {
        let predicted_risk = predict_risk(&graph, test_patient.clone(), threshold);
        let predicted_label = if predicted_risk >= 0.5 { 1 } else { 0 };

        if predicted_label == test_patient.HadHeartAttack_INT {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / test_data.len() as f64;
    println!("Model accuracy for selected quarter: {:.2}%", accuracy * 100.0);

    Ok(())
}