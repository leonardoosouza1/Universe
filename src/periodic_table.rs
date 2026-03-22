use bevy::prelude::*;

pub struct ElementData {
    pub name: &'static str,
    pub symbol: &'static str,
    pub atomic_number: u32,
    pub mass: f32,
    pub electronegativity: f32,
    pub radius: f32,
    pub color: Color,
    pub valence: usize,
}

// DEFINITIVE IARA PERIODIC TABLE v2101
pub const PERIODIC_TABLE: &[ElementData] = &[
    // 1-2: Período 1
    ElementData { name: "Hydrogen", symbol: "H", atomic_number: 1, mass: 1.008, electronegativity: 2.20, radius: 37.0, color: Color::srgb(0.7, 0.7, 1.0), valence: 1 },
    ElementData { name: "Helium", symbol: "He", atomic_number: 2, mass: 4.002, electronegativity: 0.00, radius: 31.0, color: Color::srgb(0.8, 1.0, 1.0), valence: 0 },
    
    // 3-10: Período 2
    ElementData { name: "Lithium", symbol: "Li", atomic_number: 3, mass: 6.941, electronegativity: 0.98, radius: 152.0, color: Color::srgb(0.8, 0.3, 0.6), valence: 1 },
    ElementData { name: "Beryllium", symbol: "Be", atomic_number: 4, mass: 9.012, electronegativity: 1.57, radius: 112.0, color: Color::srgb(0.5, 0.7, 0.5), valence: 2 },
    ElementData { name: "Boron", symbol: "B", atomic_number: 5, mass: 10.811, electronegativity: 2.04, radius: 88.0, color: Color::srgb(1.0, 0.8, 0.7), valence: 3 },
    ElementData { name: "Carbon", symbol: "C", atomic_number: 6, mass: 12.011, electronegativity: 2.55, radius: 77.0, color: Color::srgb(0.2, 0.2, 0.2), valence: 4 },
    ElementData { name: "Nitrogen", symbol: "N", atomic_number: 7, mass: 14.007, electronegativity: 3.04, radius: 70.0, color: Color::srgb(0.5, 0.5, 1.0), valence: 5 },
    ElementData { name: "Oxygen", symbol: "O", atomic_number: 8, mass: 15.999, electronegativity: 3.44, radius: 66.0, color: Color::srgb(1.0, 0.2, 0.2), valence: 6 },
    ElementData { name: "Fluorine", symbol: "F", atomic_number: 9, mass: 18.998, electronegativity: 3.98, radius: 64.0, color: Color::srgb(0.3, 1.0, 0.4), valence: 7 },
    ElementData { name: "Neon", symbol: "Ne", atomic_number: 10, mass: 20.180, electronegativity: 0.00, radius: 58.0, color: Color::srgb(1.0, 0.5, 0.0), valence: 0 },
    
    // 11-18: Período 3
    ElementData { name: "Sodium", symbol: "Na", atomic_number: 11, mass: 22.990, electronegativity: 0.93, radius: 186.0, color: Color::srgb(0.5, 0.2, 0.8), valence: 1 },
    ElementData { name: "Magnesium", symbol: "Mg", atomic_number: 12, mass: 24.305, electronegativity: 1.31, radius: 160.0, color: Color::srgb(0.6, 0.6, 0.2), valence: 2 },
    ElementData { name: "Aluminum", symbol: "Al", atomic_number: 13, mass: 26.982, electronegativity: 1.61, radius: 143.0, color: Color::srgb(0.7, 0.7, 0.7), valence: 3 },
    ElementData { name: "Silicon", symbol: "Si", atomic_number: 14, mass: 28.085, electronegativity: 1.90, radius: 118.0, color: Color::srgb(0.4, 0.4, 0.6), valence: 4 },
    ElementData { name: "Phosphorus", symbol: "P", atomic_number: 15, mass: 30.974, electronegativity: 2.19, radius: 110.0, color: Color::srgb(1.0, 0.6, 0.0), valence: 5 },
    ElementData { name: "Sulfur", symbol: "S", atomic_number: 16, mass: 32.065, electronegativity: 2.58, radius: 104.0, color: Color::srgb(1.0, 1.0, 0.2), valence: 6 },
    ElementData { name: "Chlorine", symbol: "Cl", atomic_number: 17, mass: 35.453, electronegativity: 3.16, radius: 99.0, color: Color::srgb(0.1, 0.8, 0.1), valence: 7 },
    ElementData { name: "Argon", symbol: "Ar", atomic_number: 18, mass: 39.948, electronegativity: 0.00, radius: 94.0, color: Color::srgb(1.0, 0.2, 1.0), valence: 0 },
    
    // 19-36: Período 4
    ElementData { name: "Potassium", symbol: "K", atomic_number: 19, mass: 39.098, electronegativity: 0.82, radius: 227.0, color: Color::srgb(0.8, 0.1, 0.1), valence: 1 },
    ElementData { name: "Calcium", symbol: "Ca", atomic_number: 20, mass: 40.078, electronegativity: 1.00, radius: 197.0, color: Color::srgb(0.6, 0.4, 0.1), valence: 2 },
    ElementData { name: "Scandium", symbol: "Sc", atomic_number: 21, mass: 44.956, electronegativity: 1.36, radius: 162.0, color: Color::srgb(0.9, 0.9, 0.9), valence: 2 },
    ElementData { name: "Titanium", symbol: "Ti", atomic_number: 22, mass: 47.867, electronegativity: 1.54, radius: 147.0, color: Color::srgb(0.7, 0.7, 0.7), valence: 2 },
    ElementData { name: "Vanadium", symbol: "V", atomic_number: 23, mass: 50.942, electronegativity: 1.63, radius: 134.0, color: Color::srgb(0.6, 0.6, 0.6), valence: 2 },
    ElementData { name: "Chromium", symbol: "Cr", atomic_number: 24, mass: 51.996, electronegativity: 1.66, radius: 128.0, color: Color::srgb(0.5, 0.6, 0.8), valence: 1 },
    ElementData { name: "Manganese", symbol: "Mn", atomic_number: 25, mass: 54.938, electronegativity: 1.55, radius: 127.0, color: Color::srgb(0.6, 0.5, 0.7), valence: 2 },
    ElementData { name: "Iron", symbol: "Fe", atomic_number: 26, mass: 55.845, electronegativity: 1.83, radius: 126.0, color: Color::srgb(0.8, 0.4, 0.2), valence: 2 },
    ElementData { name: "Cobalt", symbol: "Co", atomic_number: 27, mass: 58.933, electronegativity: 1.88, radius: 125.0, color: Color::srgb(0.2, 0.4, 0.8), valence: 2 },
    ElementData { name: "Nickel", symbol: "Ni", atomic_number: 28, mass: 58.693, electronegativity: 1.91, radius: 124.0, color: Color::srgb(0.3, 0.7, 0.5), valence: 2 },
    ElementData { name: "Copper", symbol: "Cu", atomic_number: 29, mass: 63.546, electronegativity: 1.90, radius: 128.0, color: Color::srgb(0.9, 0.5, 0.3), valence: 1 },
    ElementData { name: "Zinc", symbol: "Zn", atomic_number: 30, mass: 65.380, electronegativity: 1.65, radius: 134.0, color: Color::srgb(0.5, 0.5, 0.6), valence: 2 },
    ElementData { name: "Gallium", symbol: "Ga", atomic_number: 31, mass: 69.723, electronegativity: 1.81, radius: 135.0, color: Color::srgb(0.7, 0.6, 0.6), valence: 3 },
    ElementData { name: "Germanium", symbol: "Ge", atomic_number: 32, mass: 72.630, electronegativity: 2.01, radius: 122.0, color: Color::srgb(0.4, 0.6, 0.4), valence: 4 },
    ElementData { name: "Arsenic", symbol: "As", atomic_number: 33, mass: 74.922, electronegativity: 2.18, radius: 115.0, color: Color::srgb(0.7, 0.5, 0.8), valence: 5 },
    ElementData { name: "Selenium", symbol: "Se", atomic_number: 34, mass: 78.971, electronegativity: 2.55, radius: 103.0, color: Color::srgb(1.0, 0.6, 0.0), valence: 6 },
    ElementData { name: "Bromine", symbol: "Br", atomic_number: 35, mass: 79.904, electronegativity: 2.96, radius: 114.0, color: Color::srgb(0.6, 0.2, 0.2), valence: 7 },
    ElementData { name: "Krypton", symbol: "Kr", atomic_number: 36, mass: 83.798, electronegativity: 3.00, radius: 109.0, color: Color::srgb(0.4, 0.8, 0.8), valence: 0 },

    // 37-54: Período 5
    ElementData { name: "Rubidium", symbol: "Rb", atomic_number: 37, mass: 85.468, electronegativity: 0.82, radius: 248.0, color: Color::srgb(0.4, 0.2, 0.6), valence: 1 },
    ElementData { name: "Strontium", symbol: "Sr", atomic_number: 38, mass: 87.620, electronegativity: 0.95, radius: 215.0, color: Color::srgb(0.0, 0.8, 0.0), valence: 2 },
    ElementData { name: "Yttrium", symbol: "Y", atomic_number: 39, mass: 88.906, electronegativity: 1.22, radius: 180.0, color: Color::srgb(0.6, 1.0, 1.0), valence: 2 },
    ElementData { name: "Zirconium", symbol: "Zr", atomic_number: 40, mass: 91.224, electronegativity: 1.33, radius: 160.0, color: Color::srgb(0.6, 0.9, 0.9), valence: 2 },
    ElementData { name: "Niobium", symbol: "Nb", atomic_number: 41, mass: 92.906, electronegativity: 1.60, radius: 146.0, color: Color::srgb(0.4, 0.8, 0.7), valence: 1 },
    ElementData { name: "Molybdenum", symbol: "Mo", atomic_number: 42, mass: 95.950, electronegativity: 2.16, radius: 139.0, color: Color::srgb(0.3, 0.7, 0.6), valence: 1 },
    ElementData { name: "Technetium", symbol: "Tc", atomic_number: 43, mass: 98.000, electronegativity: 1.90, radius: 136.0, color: Color::srgb(0.2, 0.5, 0.5), valence: 1 },
    ElementData { name: "Ruthenium", symbol: "Ru", atomic_number: 44, mass: 101.070, electronegativity: 2.20, radius: 134.0, color: Color::srgb(0.1, 0.6, 0.6), valence: 1 },
    ElementData { name: "Rhodium", symbol: "Rh", atomic_number: 45, mass: 102.906, electronegativity: 2.28, radius: 134.0, color: Color::srgb(0.0, 0.5, 0.5), valence: 1 },
    ElementData { name: "Palladium", symbol: "Pd", atomic_number: 46, mass: 106.420, electronegativity: 2.20, radius: 137.0, color: Color::srgb(0.0, 0.4, 0.5), valence: 0 },
    ElementData { name: "Silver", symbol: "Ag", atomic_number: 47, mass: 107.868, electronegativity: 1.93, radius: 144.0, color: Color::srgb(0.9, 0.9, 0.9), valence: 1 },
    ElementData { name: "Cadmium", symbol: "Cd", atomic_number: 48, mass: 112.411, electronegativity: 1.69, radius: 151.0, color: Color::srgb(1.0, 0.8, 0.6), valence: 2 },
    ElementData { name: "Indium", symbol: "In", atomic_number: 49, mass: 114.818, electronegativity: 1.78, radius: 167.0, color: Color::srgb(0.6, 0.5, 0.5), valence: 3 },
    ElementData { name: "Tin", symbol: "Sn", atomic_number: 50, mass: 118.710, electronegativity: 1.96, radius: 140.0, color: Color::srgb(0.4, 0.5, 0.6), valence: 4 },
    ElementData { name: "Antimony", symbol: "Sb", atomic_number: 51, mass: 121.760, electronegativity: 2.05, radius: 140.0, color: Color::srgb(0.6, 0.4, 0.8), valence: 5 },
    ElementData { name: "Tellurium", symbol: "Te", atomic_number: 52, mass: 127.600, electronegativity: 2.10, radius: 142.0, color: Color::srgb(0.8, 0.4, 0.0), valence: 6 },
    ElementData { name: "Iodine", symbol: "I", atomic_number: 53, mass: 126.904, electronegativity: 2.66, radius: 133.0, color: Color::srgb(0.4, 0.0, 0.4), valence: 7 },
    ElementData { name: "Xenon", symbol: "Xe", atomic_number: 54, mass: 131.293, electronegativity: 2.60, radius: 130.0, color: Color::srgb(0.3, 0.6, 0.7), valence: 0 },

    // 55-86: Período 6
    ElementData { name: "Cesium", symbol: "Cs", atomic_number: 55, mass: 132.905, electronegativity: 0.79, radius: 265.0, color: Color::srgb(0.3, 0.1, 0.5), valence: 1 },
    ElementData { name: "Barium", symbol: "Ba", atomic_number: 56, mass: 137.327, electronegativity: 0.89, radius: 222.0, color: Color::srgb(0.0, 0.6, 0.0), valence: 2 },
    ElementData { name: "Gold", symbol: "Au", atomic_number: 79, mass: 196.967, electronegativity: 2.54, radius: 144.0, color: Color::srgb(1.0, 0.8, 0.0), valence: 1 },
    ElementData { name: "Mercury", symbol: "Hg", atomic_number: 80, mass: 200.592, electronegativity: 2.00, radius: 151.0, color: Color::srgb(0.7, 0.7, 0.8), valence: 2 },
    ElementData { name: "Lead", symbol: "Pb", atomic_number: 82, mass: 207.200, electronegativity: 2.33, radius: 175.0, color: Color::srgb(0.3, 0.3, 0.4), valence: 4 },
    
    // 87-118: Período 7
    ElementData { name: "Uranium", symbol: "U", atomic_number: 92, mass: 238.029, electronegativity: 1.38, radius: 156.0, color: Color::srgb(0.2, 1.0, 0.2), valence: 2 },
    ElementData { name: "Oganesson", symbol: "Og", atomic_number: 118, mass: 294.000, electronegativity: 0.00, radius: 157.0, color: Color::srgb(1.0, 0.0, 0.5), valence: 0 },
];

pub const METAL_ELEMENTS: &[&str] = &[
    "Lithium", "Sodium", "Potassium", "Rubidium", "Cesium", "Francium",
    "Beryllium", "Magnesium", "Calcium", "Strontium", "Barium", "Radium",
    "Iron", "Copper", "Silver", "Gold", "Mercury", "Aluminum", "Lead"
];

pub const NON_METAL_ELEMENTS: &[&str] = &[
    "Hydrogen", "Carbon", "Nitrogen", "Oxygen", "Phosphorus", "Sulfur", "Selenium"
];

pub const NOBLE_GASES: &[&str] = &[
    "Helium", "Neon", "Argon", "Krypton", "Xenon", "Radon", "Oganesson"
];

pub const HALOGENS: &[&str] = &[
    "Fluorine", "Chlorine", "Bromine", "Iodine", "Astatine", "Tennessine"
];

pub const METALLOIDS: &[&str] = &[
    "Boron", "Silicon", "Germanium", "Arsenic", "Antimony", "Tellurium", "Polonium"
];

pub fn get_element_by_name(name: &str) -> &'static ElementData {
    PERIODIC_TABLE.iter().find(|e| e.name == name).unwrap_or(&PERIODIC_TABLE[0])
}

pub fn get_elements_by_category(category: &[&str]) -> Vec<&'static ElementData> {
    PERIODIC_TABLE
        .iter()
        .filter(|e| category.contains(&e.name))
        .collect()
}
