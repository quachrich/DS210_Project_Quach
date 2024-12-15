use petgraph::graph::{Graph, NodeIndex};
use serde::Deserialize;
use rand::seq::SliceRandom;

#[derive(Debug, Deserialize, Clone)]
#[derive(Default)]
//Stores all value of CSV 
pub struct Patient {
    pub State_INT: u8,
    pub Sex_INT: u8,
    pub GeneralHealth_INT: u8,
    pub PhysicalHealthDays: u8,
    pub MentalHealthDays: u8,
    pub LastCheckupTime_INT: u8,
    pub PhysicalActivities_INT: u8,
    pub SleepHours: u8,
    pub RemovedTeeth_INT: u8,
    pub HadHeartAttack_INT: u8,
    pub HadAngina_INT: u8,
    pub HadStroke_INT: u8,
    pub HadAsthma_INT: u8,
    pub HadSkinCancer_INT: u8,
    pub HadCOPD_INT: u8,
    pub HadDepressiveDisorder_INT: u8,
    pub HadKidneyDisease_INT: u8,
    pub HadArthritis_INT: u8,
    pub HadDiabetes_INT: u8,
    pub DeafOrHardOfHearing_INT: u8,
    pub BlindOrVisionDifficulty_INT: u8,
    pub DifficultyConcentrating_INT: u8,
    pub DifficultyWalking_INT: u8,
    pub DifficultyDressingBathing_INT: u8,
    pub DifficultyErrands_INT: u8,
    pub SmokerStatus_INT: u8,
    pub ECigaretteUsage_INT: u8,
    pub ChestScan_INT: u8,
    pub RaceEthnicityCategory_INT: u8,
    pub AgeCategory_INT: u8,
    pub HeightInMeters: f32,
    pub WeightInKilograms: f32,
    pub BMI: f32,
    pub AlcoholDrinkers_INT: u8,
    pub HIVTesting_INT: u8,
    pub FluVaxLast12_INT: u8,
    pub PneumoVaxEver_INT: u8,
    pub TetanusLast10Tdap_INT: u8,
    pub HighRiskLastYear_INT: u8,
}


//Loads data set
pub fn load_dataset(file_path: &str) -> Result<Vec<Patient>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;
    let mut dataset = Vec::new();

    for (i, result) in reader.deserialize().enumerate() {
        //Error handling
        match result {
            Ok(patient) => dataset.push(patient),
            Err(err) => eprintln!("Error on record {}: {:?}", i + 1, err),
        }
    }

    Ok(dataset)
}

pub fn randomize_and_split(data: Vec<Patient>, test_ratio: f64) -> ((Vec<Patient>, Vec<Patient>), (Vec<Patient>, Vec<Patient>), (Vec<Patient>, Vec<Patient>), (Vec<Patient>, Vec<Patient>)) {
    //Shuffles the data
    let mut rng = rand::thread_rng();
    let mut shuffled_data = data.clone();
    shuffled_data.shuffle(&mut rng);

    let quarter_size = shuffled_data.len() / 50;
    //Divide into 50 different sections each quarter contains one slice of /50 until the 4th one which contains leftovers
    let quarters = vec![
        &shuffled_data[..quarter_size],
        &shuffled_data[quarter_size..(2 * quarter_size)],
        &shuffled_data[(2 * quarter_size)..(3 * quarter_size)],
        &shuffled_data[(3 * quarter_size)..],
    ];
    
    //Code is split into training and testing 
    let splits: Vec<(Vec<Patient>, Vec<Patient>)> = quarters
        .iter()
        .map(|quarter| {
            let test_size = (quarter.len() as f64 * test_ratio).ceil() as usize;
            let train_data = quarter[..quarter.len() - test_size].to_vec();
            let test_data = quarter[quarter.len() - test_size..].to_vec();
            (train_data, test_data)
        })
        .collect();

    (
        splits[0].clone(),
        splits[1].clone(),
        splits[2].clone(),
        splits[3].clone(),
    )
}

//Builds a graph nodes equals a patient
pub fn build_graph(patients: &[Patient], threshold: u32) -> Graph<Patient, u32> {
    let mut graph = Graph::new();

    let node_indices: Vec<_> = patients
        .iter()
        .map(|p| graph.add_node(p.clone()))
        .collect();

    for i in 0..patients.len() {
        for j in (i + 1)..patients.len() {
            //Similarity adds edge based off threshold
            let similarity = calculate_similarity(&patients[i], &patients[j]);
            if similarity >= threshold {
                graph.add_edge(node_indices[i], node_indices[j], similarity);
            }
        }
    }

    graph
}

//equals(==) diff low (3) diff med (5) diff high (10)
pub fn calculate_similarity(p1: &Patient, p2: &Patient) -> u32 {
    let mut score = 0;
    if p1.State_INT == p2.State_INT { score += 1; }
    if p1.Sex_INT == p2.Sex_INT { score += 1; }
    if (p1.GeneralHealth_INT as i8 - p2.GeneralHealth_INT as i8).abs() <= 3 { score += 1; }
    if (p1.PhysicalActivities_INT as i8 - p2.PhysicalActivities_INT as i8).abs() <= 3 { score += 1; }
    if (p1.MentalHealthDays as i8 - p2.MentalHealthDays as i8).abs() <= 3 { score += 1; }
    if (p1.LastCheckupTime_INT as i8 - p2.LastCheckupTime_INT as i8).abs() <= 3 { score += 1; }
    if (p1.PhysicalHealthDays as i8 - p2.PhysicalHealthDays as i8).abs() <= 3 { score += 1; }
    if (p1.SleepHours as i8 - p2.SleepHours as i8).abs() <= 3 { score += 1; }
    if p1.RaceEthnicityCategory_INT == p2.RaceEthnicityCategory_INT { score += 1; }
    if (p1.WeightInKilograms - p2.WeightInKilograms).abs() <= 10.0 { score += 1; }
    if (p1.BMI - p2.BMI).abs() <= 5.0 { score += 1; }
    score
    //Score is what is going to be use to know similarity 
}

pub fn predict_risk(graph: &Graph<Patient, u32>, new_patient: Patient, threshold: u32) -> f64 {
    let mut graph = graph.clone();
    let new_node = graph.add_node(new_patient.clone());

    //Adds node for each patient
    for i in 0..graph.node_count() - 1 {
        let existing_patient = graph.node_weight(NodeIndex::new(i)).unwrap();
        let similarity = calculate_similarity(existing_patient, &new_patient);
        if similarity >= threshold {
            graph.add_edge(new_node, NodeIndex::new(i), similarity);
        }
    }

    let neighbors = graph.neighbors(new_node);
    let mut risk_sum = 0.0;
    let mut count = 0;

    
    for neighbor in neighbors {
        let neighbor_patient = graph.node_weight(neighbor).unwrap();
        risk_sum += neighbor_patient.HadHeartAttack_INT as f64;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
//Risk score 
    risk_sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_graph() {
        // Made up patient data
        let patients = vec![
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, SleepHours: 7, ..Default::default() },
            Patient { State_INT: 1, Sex_INT: 0, GeneralHealth_INT: 3, SleepHours: 7, ..Default::default() },
            Patient { State_INT: 2, Sex_INT: 1, GeneralHealth_INT: 2, SleepHours: 8, ..Default::default() },
        ];

        // Test graph creation
        let graph = build_graph(&patients, 1);
        assert_eq!(graph.node_count(), 3, "Graph should have 3 nodes");
        assert!(graph.edge_count() >= 1, "Graph should have at least 1 edge");
    }
}

    #[test]
    fn test_predict_risk() {
        //Made up patient data
        let patient1 = Patient { HadHeartAttack_INT: 1, ..Default::default() };
        let patient2 = Patient { HadHeartAttack_INT: 0, ..Default::default() };
        let new_patient = Patient { HadHeartAttack_INT: 0, ..Default::default() };

        let mut graph = Graph::<Patient, u32>::new();
        let node1 = graph.add_node(patient1);
        let node2 = graph.add_node(patient2);
        //Test Predict Risk Function
        graph.add_edge(node1, node2, 5);
        let risk = predict_risk(&graph, new_patient, 1);
        assert_eq!(risk, 0.5, "Predicted risk should be 0.5");
    }
