use bevy::prelude::*;

pub struct ElementData {
    pub name: &'static str,
    pub symbol: &'static str,
    pub mass: f32,
    pub electronegativity: f32,
    pub radius: f32,
    pub color: Color,
}

pub const PERIODIC_TABLE: &[ElementData] = &[
    ElementData { name: "Hydrogen", symbol: "H", mass: 1.008, electronegativity: 2.20, radius: 37.0, color: Color::srgb(0.7, 0.7, 1.0) },
    ElementData { name: "Deuterium", symbol: "D", mass: 2.014, electronegativity: 2.20, radius: 37.0, color: Color::srgb(0.5, 0.5, 1.0) },
    ElementData { name: "Helium3", symbol: "He3", mass: 3.016, electronegativity: 0.00, radius: 31.0, color: Color::srgb(1.0, 0.8, 0.2) },
    ElementData { name: "Helium", symbol: "He", mass: 4.002, electronegativity: 0.00, radius: 31.0, color: Color::srgb(1.0, 1.0, 0.5) },
    ElementData { name: "Lithium", symbol: "Li", mass: 6.941, electronegativity: 0.98, radius: 152.0, color: Color::srgb(0.8, 0.3, 0.6) },
    ElementData { name: "Carbon", symbol: "C", mass: 12.011, electronegativity: 2.55, radius: 77.0, color: Color::srgb(0.2, 0.2, 0.2) },
    ElementData { name: "Oxygen", symbol: "O", mass: 15.999, electronegativity: 3.44, radius: 66.0, color: Color::srgb(1.0, 0.2, 0.2) },
    ElementData { name: "Silicon", symbol: "Si", mass: 28.085, electronegativity: 1.90, radius: 118.0, color: Color::srgb(0.4, 0.4, 0.6) },
    ElementData { name: "Iron", symbol: "Fe", mass: 55.845, electronegativity: 1.83, radius: 126.0, color: Color::srgb(0.8, 0.4, 0.2) },
    ElementData { name: "Gold", symbol: "Au", mass: 196.967, electronegativity: 2.54, radius: 144.0, color: Color::srgb(1.0, 0.8, 0.0) },
    ElementData { name: "Uranium", symbol: "U", mass: 238.029, electronegativity: 1.38, radius: 138.0, color: Color::srgb(0.2, 1.0, 0.2) },
];

pub fn get_element_by_name(name: &str) -> &'static ElementData {
    PERIODIC_TABLE.iter().find(|e| e.name == name).unwrap_or(&PERIODIC_TABLE[0])
}

pub fn get_element_color(name: &str) -> Color {
    get_element_by_name(name).color
}
