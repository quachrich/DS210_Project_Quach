use petgraph::graph::{Graph, NodeIndex};
use serde::Deserialize;
use rand::seq::SliceRandom;


#[serde(rename_all = "PascalCase")]

//All the tested CSV variables

#[derive(Debug, Deserialize, Clone)]
pub struct Patient {
    #[serde(rename = "State_INT")]
    pub state_int: u8,
    pub Sex_INT: u8,
    pub general_health_int: u8,
    pub physical_health_days: u8,
    pub mental_health_days: u8,
    pub last_checkup_time_int: u8,
    pub physical_activities_int: u8,
    pub sleep_hours: u8,
    #[serde(rename = "RaceEthnicityCategory_INT")]
    pub race_ethnicity_category_int: u8,
    pub age_category_int: u8,
    pub weight_in_kilograms: f32,
    pub bmi: f32,
    #[serde(rename = "RemovedTeeth_INT")]
    pub removed_teeth: u8,
    #[serde(rename = "HadHeartAttack_INT")]
    pub had_heart_attack_int: u8,
    #[serde(rename = "HadAngina_INT")]
    pub had_angina_int: u8,
    #[serde(rename = "HadStroke_INT")]
    pub had_stroke_int: u8,
    #[serde(rename = "HadSkinCancer_INT")]
    pub had_skin_cancer: u8,
    #[serde(rename = "HadCOPD_INT")]
    pub had_copd_int: u8,
    #[serde(rename = "HadDepressiveDisorder_INT")]
    pub had_depressive_disorder_int: u8,
    #[serde(rename = "HadKidneyDisease_INT")]
    pub had_kidney_disease_int: u8,
    #[serde(rename = "HadArthritis_INT")]
    pub had_arthritis_int: u8,
    #[serde(rename = "HadDiabetes_INT")]
    pub had_diabetes_int: u8,
    #[serde(rename = "DeafOrHardOfHearing_INT")]
    pub deaf_or_hard_of_hearing_int: u8,
    #[serde(rename = "BlindOrVisionDifficulty_INT")]
    pub blind_or_vision_difficulty_int: u8,
    #[serde(rename = "DifficultyConcentrating_INT")]
    pub difficulty_concentrating_int: u8,
    #[serde(rename = "DifficultyWalking_INT")]
    pub difficulty_walking_int: u8,
    #[serde(rename = "DifficultyDressingBathing_INT")]
    pub difficulty_dressing_bathing_int: u8,
    #[serde(rename = "DifficultyErrands_INT")]
    pub difficulty_errands_int: u8,
    #[serde(rename = "SmokerStatus_INT")]
    pub smoker_status_int: u8,
    #[serde(rename = "ECigaretteUsage_INT")]
    pub e_cigarette_usage_int: u8,
    #[serde(rename = "ChestScan_INT")]
    pub chest_scan_int: u8,
    #[serde(rename = "AlcoholDrinkers_INT")]
    pub alcohol_drinkers_int: u8,
    #[serde(rename = "HIVTesting_INT")]
    pub hiv_testing_int: u8,
    #[serde(rename = "FluVaxLast12_INT")]
    pub flu_vax_last_12_int: u8,
    #[serde(rename = "PneumoVacEver_INT")]
    pub pneumo_vac_ever_int: u8,
    #[serde(rename = "TetanusLast10Tdap_INT")]
    pub tetanus_last_10_tdap_int: u8,
    #[serde(rename = "HighRiskLastYear_INT")]
    pub high_risk_last_year_int: u8,
    #[serde(rename = "CovidPos_INT")]
    pub covid_pos_int: u8,
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
    if (p1.state_int as i8 == p2.state_int as i8) { score += 1; }//equal
    if (p1.Sex_INT as i8 == p2.Sex_INT as i8) { score += 1; }//equal
    if (p1.general_health_int as i8 - p2.general_health_int as i8).abs() <= 3 { score += 1; }//small
    if (p1.physical_activities_int as i8 - p2.physical_activities_int as i8).abs() <= 3 { score += 1; }//small
    if (p1.mental_health_days as i8 - p2.mental_health_days as i8).abs() <= 3 { score += 1; }//small
    if (p1.last_checkup_time_int as i8 - p2.last_checkup_time_int as i8).abs() <= 3 { score += 1; }//small
    if (p1.physical_health_days as i8 - p2.physical_health_days as i8).abs() <= 3 { score += 1; }//small
    if (p1.sleep_hours as i8 - p2.sleep_hours as i8).abs() <= 3 {score += 1} //small
    if (p1.race_ethnicity_category_int as i8 == p2.race_ethnicity_category_int as i8) { score += 1; }//equal
    if (p1.weight_in_kilograms as i8 - p2.weight_in_kilograms as i8).abs() <= 10 { score += 1; }//large
    if (p1.bmi as i8 - p2.bmi as i8).abs() <= 5 { score += 1; }//medium
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
        risk_sum += neighbor_patient.had_heart_attack_int as f64;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    risk_sum / count as f64
}
