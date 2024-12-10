use petgraph::graph::{Graph, NodeIndex};
use serde::Deserialize;
use rand::seq::SliceRandom;

// All the tested CSV variables
#[derive(Debug, Deserialize, Clone)]

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
    pub CovidPos_INT: u8,
}


//State_INT,Sex_INT,GeneralHealth_INT,PhysicalHealthDays,MentalHealthDays,LastCheckupTime_INT,
//PhysicalActivities_INT,SleepHours,RemovedTeeth_INT,HadHeartAttack_INT,HadAngina_INT,
//HadStroke_INT,HadAsthma_INT,HadSkinCancer_INT,HadCOPD_INT,HadDepressiveDisorder_INT,HadKidneyDisease_INT,
//HadArthritis_INT,HadDiabetes_INT,DeafOrHardOfHearing_INT,BlindOrVisionDifficulty_INT,DifficultyConcentrating_INT,
//DifficultyWalking_INT,DifficultyDressingBathing_INT,DifficultyErrands_INT,SmokerStatus_INT,ECigaretteUsage_INT,
//ChestScan_INT,RaceEthnicityCategory_INT,AgeCategory_INT,HeightInMeters,WeightInKilograms,BMI,AlcoholDrinkers_INT,
//HIVTesting_INT,FluVaxLast12_INT,PneumoVaxEver_INT,TetanusLast10Tdap_INT,HighRiskLastYear_INT,CovidPos_INT

//Loading Dataset
pub fn load_dataset(file_path: &str) -> Result<Vec<Patient>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;
    println!("CSV Headers: {:?}", reader.headers()?);

    let mut dataset = Vec::new();

    for (i, result) in reader.deserialize().enumerate() {
        match result {
            Ok(patient) => dataset.push(patient),
            Err(err) => eprintln!("Error on record {}: {:?}", i + 1, err),
        }
    }

    Ok(dataset)
}


pub fn split_and_randomize(data: Vec<Patient>) -> (Vec<Patient>, Vec<Patient>, Vec<Patient>, Vec<Patient>) {
    let mut rng = rand::thread_rng();
    let mut shuffled_data = data.clone();
    
    // Randomize the dataset
    shuffled_data.shuffle(&mut rng);
    
    // Calculate the size for each 50 different sections
    let quarter_size = shuffled_data.len() / 50;
    
    // Split the dataset into four quarters
    let first_quarter = shuffled_data[..quarter_size].to_vec();
    let second_quarter = shuffled_data[quarter_size..(2 * quarter_size)].to_vec();
    let third_quarter = shuffled_data[(2 * quarter_size)..(3 * quarter_size)].to_vec();
    let fourth_quarter = shuffled_data[(3 * quarter_size)..].to_vec();

    (first_quarter, second_quarter, third_quarter, fourth_quarter)
}

pub fn split_dataset(data: Vec<Patient>, test_ratio: f64) -> (Vec<Patient>, Vec<Patient>) {
    let mut rng = rand::thread_rng();
    let mut shuffled_data = data.clone();
    
    // Randomize the dataset
    shuffled_data.shuffle(&mut rng);

    // Determine the size of the test set
    let test_size = (shuffled_data.len() as f64 * test_ratio).ceil() as usize;

    // Split into test and training sets
    let test_data = shuffled_data.split_off(shuffled_data.len() - test_size);
    let train_data = shuffled_data;

    (train_data, test_data)
}

pub fn process_dataset(
    data: Vec<Patient>, 
    test_ratio: f64
) -> (
    (Vec<Patient>, Vec<Patient>), 
    (Vec<Patient>, Vec<Patient>), 
    (Vec<Patient>, Vec<Patient>), 
    (Vec<Patient>, Vec<Patient>)
) {
    // Split and randomize the dataset into quarters
    let (q1, q2, q3, q4) = split_and_randomize(data);

    // Further split each quarter into training and testing datasets
    let first_split = split_dataset(q1, test_ratio);
    let second_split = split_dataset(q2, test_ratio);
    let third_split = split_dataset(q3, test_ratio);
    let fourth_split = split_dataset(q4, test_ratio);

    (first_split, second_split, third_split, fourth_split)
}

pub fn build_graph(patients: &[Patient], threshold: u32) -> Graph<Patient, u32> {
    let mut graph = Graph::new();

    let node_indices: Vec<_> = patients
        .iter()
        .map(|p| graph.add_node(p.clone()))
        .collect();

    for i in 0..patients.len() {
        for j in (i + 1)..patients.len() {
            let similarity = calculate_similarity(&patients[i], &patients[j]);
            if similarity >= threshold {
                graph.add_edge(node_indices[i], node_indices[j], similarity);
            }
        }
    }

    graph
}

//Broken up into 4 different similarity scores equal(==), small(3), medium(5), high(10)
pub fn calculate_similarity(p1: &Patient, p2: &Patient) -> u32 {
    let mut score = 0;
    if (p1.State_INT as i8 == p2.State_INT as i8) { score += 1; }//equal
    if (p1.Sex_INT as i8 == p2.Sex_INT as i8) { score += 1; }//equal
    if (p1.GeneralHealth_INT as i8 - p2.GeneralHealth_INT as i8).abs() <= 3 { score += 1; }//small
    if (p1.PhysicalActivities_INT as i8 - p2.PhysicalActivities_INT as i8).abs() <= 3 { score += 1; }//small
    if (p1.MentalHealthDays as i8 - p2.MentalHealthDays as i8).abs() <= 3 { score += 1; }//small
    if (p1.LastCheckupTime_INT as i8 - p2.LastCheckupTime_INT as i8).abs() <= 3 { score += 1; }//small
    if (p1.PhysicalHealthDays as i8 - p2.PhysicalHealthDays as i8).abs() <= 3 { score += 1; }//small
    if (p1.SleepHours as i8 - p2.SleepHours as i8).abs() <= 3 {score += 1} //small
    if (p1.RaceEthnicityCategory_INT as i8 == p2.RaceEthnicityCategory_INT as i8) { score += 1; }//equal
    if (p1.WeightInKilograms as i8 - p2.WeightInKilograms as i8).abs() <= 10 { score += 1; }//large
    if (p1.BMI as i8 - p2.BMI as i8).abs() <= 5 { score += 1; }//medium
    score
}

pub fn predict_risk(graph: &Graph<Patient, u32>, new_patient: Patient, threshold: u32) -> f64 {
    let mut graph = graph.clone();
    let new_node = graph.add_node(new_patient.clone());

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

    risk_sum / count as f64
}
