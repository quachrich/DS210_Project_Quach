mod Graph;
use graph::{load_dataset, build_graph, predict_risk, split_dataset, Patient};
#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;

    #[test]
    fn test_load_dataset() {
        // Create a temporary CSV file for testing
        let csv_data = "\
State_INT,Sex_INT,GeneralHealth_INT,PhysicalHealthDays,MentalHealthDays,LastCheckupTime_INT,PhysicalActivities_INT,SleepHours,RemovedTeeth_INT,HadHeartAttack_INT
1,0,3,2,1,0,4,7,0,1
2,1,2,0,3,1,3,8,1,0
";
        let file_path = "test_dataset.csv";
        std::fs::write(file_path, csv_data).unwrap();

        // Load dataset
        let dataset = load_dataset(file_path).unwrap();

        // Check dataset size and values
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset[0].State_INT, 1);
        assert_eq!(dataset[1].Sex_INT, 1);

        // Clean up the temporary file
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_split_dataset() {
        // Create a mock dataset
        let dataset = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 2, Sex_INT: 1, GeneralHealth_INT: 2, PhysicalHealthDays: 0, MentalHealthDays: 3, LastCheckupTime_INT: 1, PhysicalActivities_INT: 3, SleepHours: 8, RemovedTeeth_INT: 1, HadHeartAttack_INT: 0, ..Default::default() },
            Patient { State_INT: 3, Sex_INT: 0, GeneralHealth_INT: 1, PhysicalHealthDays: 1, MentalHealthDays: 2, LastCheckupTime_INT: 2, PhysicalActivities_INT: 5, SleepHours: 6, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Split dataset
        let (train, test) = split_dataset(dataset, 0.33);

        // Check sizes
        assert_eq!(train.len(), 2);
        assert_eq!(test.len(), 1);
    }

    #[test]
    fn test_build_graph() {
        // Create a small dataset
        let patients = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Build graph
        let graph = build_graph(&patients, 2);

        // Verify nodes and edges
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1); // Similarity threshold met
    }

    #[test]
    fn test_predict_risk() {
        // Create a small dataset
        let patients = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Build graph
        let graph = build_graph(&patients, 2);

        // Predict risk for a new patient
        let new_patient = Patient {
            State_INT: 1,
            Sex_INT: 0,
            GeneralHealth_INT: 3,
            PhysicalHealthDays: 2,
            MentalHealthDays: 1,
            LastCheckupTime_INT: 0,
            PhysicalActivities_INT: 4,
            SleepHours: 7,
            RemovedTeeth_INT: 0,
            HadHeartAttack_INT: 0,
            ..Default::default()
        };

        let risk = predict_risk(&graph, new_patient, 2);

        // Check predicted risk
        assert_eq!(risk, 1.0);
    }
}
