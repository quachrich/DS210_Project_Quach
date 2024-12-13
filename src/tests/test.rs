#[cfg(test)]
mod tests {
    use DS210_Project_Quach::petgraph::{load_dataset, build_graph, predict_risk, split_dataset, Patient};
    use super::*;

    #[test]
    fn test_load_dataset() {
        let filename = "test_dataset.csv";
        let csv_data = "\
State_INT,Sex_INT,GeneralHealth_INT,PhysicalHealthDays,MentalHealthDays,LastCheckupTime_INT,PhysicalActivities_INT,SleepHours,RemovedTeeth_INT,HadHeartAttack_INT\n\
1,0,3,2,1,0,4,7,0,1\n\
2,1,2,0,3,1,3,8,1,0\n";

        // Create temporary file
        let file_path = create_temporary_csv(csv_data, filename);

        // Load dataset
        let dataset = load_dataset(&file_path).expect("Failed to load dataset");
        assert_eq!(dataset.len(), 2, "Dataset should have 2 rows");
        assert_eq!(dataset[0].State_INT, 1, "First patient's State_INT should be 1");
        assert_eq!(dataset[1].Sex_INT, 1, "Second patient's Sex_INT should be 1");

        // Clean up
        cleanup_file(&file_path);
    }

    #[test]
    fn test_split_dataset() {
        let dataset = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 2, Sex_INT: 1, GeneralHealth_INT: 2, PhysicalHealthDays: 0, MentalHealthDays: 3, LastCheckupTime_INT: 1, PhysicalActivities_INT: 3, SleepHours: 8, RemovedTeeth_INT: 1, HadHeartAttack_INT: 0, ..Default::default() },
            Patient { State_INT: 3, Sex_INT: 0, GeneralHealth_INT: 1, PhysicalHealthDays: 1, MentalHealthDays: 2, LastCheckupTime_INT: 2, PhysicalActivities_INT: 5, SleepHours: 6, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Split dataset
        let (train, test) = split_dataset(dataset, 0.33);

        assert_eq!(train.len(), 2, "Training set should have 2 patients");
        assert_eq!(test.len(), 1, "Test set should have 1 patient");
    }

    #[test]
    fn test_build_graph() {
        let patients = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Build graph
        let graph = build_graph(&patients, 2);

        assert_eq!(graph.node_count(), 2, "Graph should have 2 nodes");
        assert_eq!(graph.edge_count(), 1, "Graph should have 1 edge");
    }

    #[test]
    fn test_predict_risk() {
        let patients = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, PhysicalHealthDays: 2, MentalHealthDays: 1, LastCheckupTime_INT: 0, PhysicalActivities_INT: 4, SleepHours: 7, RemovedTeeth_INT: 0, HadHeartAttack_INT: 1, ..Default::default() },
        ];

        // Build graph
        let graph = build_graph(&patients, 2);

        // Predict risk
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
        assert_eq!(risk, 1.0, "Predicted risk should be 1.0");
    }
}
