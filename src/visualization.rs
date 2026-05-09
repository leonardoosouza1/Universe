use bevy::prelude::*;

/// Vision panel visualization for computational perception.
/// Decodes sensory_wave channels 0-63 into visual representation.
/// Shows what agent sees in real-time: NEAR zone (5-40m), FAR zone (80-200m).

#[derive(Component)]
pub struct VisionPanel {
    pub racer_entity: Entity,
}

pub struct VisionVisualizationPlugin;

impl Plugin for VisionVisualizationPlugin {
    fn build(&self, app: &mut App) {
        // Plugin currently disabled - vision data available via racer.sensory_snapshot
        // for logging and diagnostics. Full UI overlay in future release.
    }
}

/// Decode sensory_wave into human-readable spatial perception map.
/// Returns visual ASCII representation of what agent perceives.
fn decode_sensory_vision(racer: &crate::simulation::racer::Racer) -> String {
    // Channel layout:
    // ch 0-15: retina FOV bins (spatial, already pooled)
    // ch 16-23: wall distance FOV (8 angles, 3-24m)
    // ch 24-31: wall width FOV
    // ch 32-39: wall curvature FOV
    // ch 40-47: NEAR zone road ahead (5-40m, 8 distance points)
    // ch 48-55: FAR zone horizon (80-200m, 8 distance points)
    // ch 56-63: ground material + state

    let mut output = String::new();
    output.push_str("=== VISION PERCEPTION MAP ===\n");

    // RETINA (ch 0-15): FOV visualization
    output.push_str("RETINA FOV [0-15]:  ");
    for i in 0..16 {
        let val = racer.sensory_snapshot[i] as f32 / 127.0;
        let c = match val {
            v if v > 0.7 => '█',
            v if v > 0.4 => '▓',
            v if v > 0.1 => '░',
            _ => ' ',
        };
        output.push(c);
    }
    output.push('\n');

    // NEAR ZONE walls (ch 16-23): directional threat map
    output.push_str("NEAR WALLS [16-23]: ");
    for i in 16..24 {
        let val = racer.sensory_snapshot[i] as f32 / 127.0;
        let c = if val > 0.5 { '⚠' } else if val > 0.2 { '.' } else { ' ' };
        output.push(c);
    }
    output.push_str(" (L←→R)\n");

    // NEAR ZONE road (ch 40-47): 5-40m distance layering
    output.push_str("NEAR ROAD [40-47]:  ");
    for i in 40..48 {
        let val = racer.sensory_snapshot[i] as i8;
        let c = match val {
            v if v > 50 => '🟢',   // Road detected
            v if v > 0 => '🟡',    // Uncertain
            v if v < -50 => '🔴',  // Wall/obstacle
            _ => '⚫',
        };
        output.push(c);
        output.push(' ');
    }
    output.push_str("  (5m→40m)\n");

    // FAR ZONE horizon (ch 48-55): 80-200m strategic lookahead
    output.push_str("FAR HORIZON [48-55]: ");
    for i in 48..56 {
        let val = racer.sensory_snapshot[i] as i8;
        let c = match val {
            v if v > 60 => '▲',    // Crest/apex detected
            v if v > 20 => '∧',    // Gentle slope
            v if v < -60 => '▼',   // Drop/hole
            _ => '—',
        };
        output.push(c);
        output.push(' ');
    }
    output.push_str("  (80m→200m)\n");

    // Ground material (ch 56-63)
    let ground_val = racer.sensory_snapshot[56] as i8;
    let ground_type = match ground_val {
        0 => "🔷 ROAD",
        64 => "🟩 GRASS",
        100 => "🧱 WALL",
        -127 => "💥 HOLE",
        _ => "❓ UNKNOWN",
    };
    output.push_str(&format!("GROUND [56-63]:     {}\n", ground_type));

    // Summary: what vision mode is active?
    let near_threat = (16..24)
        .map(|i| racer.sensory_snapshot[i].abs() as i32)
        .sum::<i32>() as f32 / 8.0;

    let far_signal = (48..56)
        .map(|i| racer.sensory_snapshot[i].abs() as i32)
        .sum::<i32>() as f32 / 8.0;

    let vision_mode = if far_signal > near_threat * 0.5 {
        "STRATEGIC (lookahead active)"
    } else {
        "REACTIVE (immediate obstacle)"
    };

    output.push_str(&format!("\nVISION MODE: {} | NEAR_THREAT={:.1} FAR_SIGNAL={:.1}\n",
        vision_mode, near_threat, far_signal));

    output
}

/// Decode sensory vision for logging/diagnostics.
/// Future: integrate with UI overlay system.
pub fn vision_panel_text(racer: &crate::simulation::racer::Racer) -> String {
    decode_sensory_vision(racer)
}
