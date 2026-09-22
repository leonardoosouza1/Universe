/// IARA v2101 - RepoMiner (Autonomous Data Ingestion)
/// Based on Codex Part 2 & 5: Hunger Trigger

pub struct RepoMiner {
    /// Knowledge uncertainty level (0.0 to 1.0)
    pub hunger_level: f32,
    /// Threshold to trigger a "Hunt"
    pub trigger_threshold: f32,
    /// History of datasets acquired
    pub knowledge_base: Vec<String>,
}

impl RepoMiner {
    pub fn new() -> Self {
        Self {
            hunger_level: 0.0,
            trigger_threshold: 0.8,
            knowledge_base: Vec::new(),
        }
    }

    /// Update hunger based on the error/uncertainty of the engine.
    /// Axiom: "Hunger Signal (Hunger Trigger): Master LNN detects high uncertainty."
    pub fn monitor_uncertainty(&mut self, uncertainty: f32) {
        self.hunger_level += uncertainty * 0.1;

        if self.hunger_level >= self.trigger_threshold {
            self.trigger_hunt();
        }
    }

    fn trigger_hunt(&mut self) {
        let dataset_name = format!("Dataset_Physics_{}", self.knowledge_base.len());
        self.knowledge_base.push(dataset_name.clone());
        self.hunger_level = 0.0; // Satiated
    }

    /// Myelination force: Returns a boost factor if knowledge is fresh
    pub fn get_learning_boost(&self) -> f32 {
        if self.hunger_level < 0.1 {
            1.5
        } else {
            1.0
        }
    }
}
