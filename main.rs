mod graph;

use graph::{load_dataset, build_graph, predict_risk, process_dataset, split_dataset, Patient};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dataset
    let dataset = load_dataset("Heart_OnlyNum .csv")?;
    println!("Loaded {} patients.", dataset.len());

    // Process the dataset into 80% dataset
    let test_ratio = 0.2; // 20% for testing
    let (
        (train1, test1),
        _,
        _,
        _,
    ) = process_dataset(dataset, test_ratio);

    // Use only the first section
    let (train_data, test_data) = (train1, test1);

    println!(
        "Selected quarter - Training data size: {}, Testing data size: {}",
        train_data.len(),
        test_data.len()
    );

    // Build a graph from the training data
    let threshold = 2; // Similarity threshold for graph edges
    let graph = build_graph(&train_data, threshold);

    // Predict and evaluate accuracy
    let mut correct_predictions = 0;
    for test_patient in &test_data {
        let predicted_risk = predict_risk(&graph, test_patient.clone(), threshold);

        // Predict label based on risk (>= 0.5 is high risk)
        let predicted_label = if predicted_risk >= 0.5 { 1 } else { 0 };

        // Use `HadHeartAttack_INT` as the true label
        if predicted_label == test_patient.had_heart_attack_int {
            correct_predictions += 1;
        }
    }

    // Calculate and print accuracy
    let accuracy = correct_predictions as f64 / test_data.len() as f64;
    println!("Model accuracy for selected quarter: {:.2}%", accuracy * 100.0);

    Ok(())
}
